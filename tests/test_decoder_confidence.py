"""Regression checks for receiver-only confidence-guided RS repair."""
import unittest
from unittest.mock import patch

import numpy as np
from reedsolo import ReedSolomonError

from core import ecc, stc
from core.framing import MessageReader, pack_message, parse_header, unpack_body
from core.layout import frame_chunk
from core.ndca import get_positions
from core.profiles import LEGACY, PROFILES, ROBUST80
from core.video_stego import chunk_size, extract_chunk, extract_chunk_with_confidence


class ConfidenceDecoderTests(unittest.TestCase):
    def test_repairs_errors_with_confidence_including_partial_interleave_group(self):
        raw = bytes(range(200))  # Ten codewords: one full and one partial group.
        words = bytearray(ecc.interleave(ecc.encode_body(raw), reverse=True))
        scores = bytearray([200] * len(words))
        for number in (0, 8):
            for i in range(23):
                words[number*60+i] ^= i+1
                scores[number*60+i] = i
        damaged = ecc.interleave(words)
        with self.assertRaises(ecc.ECCError):
            ecc.decode_body(damaged)
        decoded, corrections, erasures = ecc.decode_body(
            damaged, list(ecc.interleave(scores)), return_erasures=True)
        self.assertEqual(decoded, raw)
        self.assertEqual(corrections, (23, 0, 0, 0, 0, 0, 0, 0, 23, 0))
        self.assertEqual(erasures, (6, 0, 0, 0, 0, 0, 0, 0, 6, 0))

    def test_hard_decoding_does_not_attempt_erasures(self):
        encoded = bytearray(ecc.encode_body(b'a' * 20))
        encoded[0] ^= 7
        with patch('core.ecc._repair_with_erasures', side_effect=AssertionError('unexpected retry')):
            raw, corrections, erasures = ecc.decode_body(encoded, [1]*60, return_erasures=True)
        self.assertEqual((raw, corrections, erasures), (b'a'*20, (1,), (0,)))

    def test_rejects_invalid_confidence(self):
        encoded = ecc.encode_body(b'a' * 20)
        for scores in ([1]*59, [1]*61, [float('nan')]*60, [-1]*60, [float('inf')]*60):
            with self.subTest(scores=scores[:1]), self.assertRaises(ValueError):
                ecc.decode_body(encoded, scores)

    def test_retries_are_bounded(self):
        with patch('core.ecc._codec') as factory:
            factory.return_value.decode.side_effect = ReedSolomonError('uncorrectable')
            with self.assertRaises(ecc.ECCError):
                ecc.decode_body(bytes(60), list(range(60)))
            self.assertEqual(factory.return_value.decode.call_count, 1 + ecc.MAX_BODY_ERASURES)

    def test_conflicting_repairs_are_rejected(self):
        with patch('core.ecc._codec') as factory:
            factory.return_value.decode.side_effect = [
                ReedSolomonError('hard failure'),
                (b'a'*20, b'a'*60, []), (b'b'*20, b'b'*60, [])]
            with self.assertRaisesRegex(ecc.ECCError, 'Ambiguous'):
                ecc.decode_body(bytes(60), list(range(60)))

    def test_body_crc_still_rejects_valid_but_wrong_rs_data(self):
        message = pack_message('x', b'payload', ROBUST80)
        header = parse_header(message[:96])
        raw, _ = ecc.decode_body(message[288:])
        changed = bytearray(raw)
        changed[1] ^= 1
        with self.assertRaisesRegex(ValueError, 'CRC32 mismatch'):
            unpack_body(header, ecc.encode_body(changed), [1]*header.body_length)

    def test_confidence_follows_stc_support_and_termination(self):
        shape = (64, 128)
        for profile in PROFILES.values():
            size = chunk_size(shape, profile)
            n = len(get_positions(shape, size*8, None if profile.stc == 'legacy' else profile.width))
            if profile.stc == 'legacy':
                H = np.zeros((size*8, n), dtype=np.uint8)
                for row in range(size*8):
                    H[row, 3*row:3*row+5] = 1
            else:
                H = stc.parity_check(size*8, profile.width)
            for weak in range(n):
                coefficients = np.full(n, 10.0*profile.delta)
                coefficients[weak] += .49*profile.delta
                with patch('core.video_stego.get_coefficient', side_effect=coefficients):
                    data, confidence = extract_chunk_with_confidence(np.zeros(shape), profile)
                expected = np.where(H[:, weak], .01*profile.delta, .5*profile.delta)
                np.testing.assert_allclose(confidence, expected.reshape(-1, 8).min(axis=1))
                self.assertEqual(data, bytes(size))

    def test_pixel_extraction_bytes_unchanged_for_all_profiles(self):
        plane = np.random.default_rng(13).integers(0, 256, (64, 128), dtype=np.uint8)
        for profile in PROFILES.values():
            data, scores = extract_chunk_with_confidence(plane, profile)
            self.assertEqual(data, extract_chunk(plane, profile))
            self.assertEqual(len(data), len(scores))
            self.assertTrue(np.all((scores >= 0) & (scores <= profile.delta/2)))

    def test_frame_header_padding_and_profile_switch_keep_confidence_aligned(self):
        shape = (180, 320)  # Multi-frame bootstrap copies, with spare byte slots.
        for profile in PROFILES.values():
            message = pack_message('x', b'payload'*7, profile)
            reader = MessageReader(profile if profile.version == 'NDV5' else LEGACY)
            total = frame = 0
            result = None
            expected_confidence = []
            while total < len(message):
                plan = frame_chunk(shape, len(message), frame, profile)
                data = message[plan.offset:plan.offset+plan.length].ljust(plan.size, b'\0')
                scores = np.arange(plan.size, dtype=float) + frame*1000
                # Only bootstrap frame padding is discarded during reading.
                consumed = plan.length if plan.offset < reader.bootstrap_size and profile != LEGACY else plan.size
                expected_confidence.extend(scores[:consumed])
                with patch('core.video_stego.extract_chunk_with_confidence', return_value=(data, scores)):
                    result = reader.feed_frame(np.zeros(shape))
                total += plan.length
                frame += 1
            self.assertEqual(result.payload, b'payload'*7)
            np.testing.assert_array_equal(reader.confidence, expected_confidence)

    def test_byte_only_reader_keeps_existing_behavior(self):
        reader = MessageReader(ROBUST80)
        result = reader.feed(pack_message('x', b'payload', ROBUST80))
        self.assertEqual(result.payload, b'payload')
        self.assertIsNone(reader.confidence)

    def test_incremental_reader_repairs_body_and_validates_complete_file(self):
        payload = bytes(range(180))
        message = pack_message('payload', payload, ROBUST80)
        words = bytearray(ecc.interleave(message[288:], reverse=True))
        scores = bytearray([200]*len(words))
        for number in (0, 8):
            for i in range(23):
                words[number*60+i] ^= i+1
                scores[number*60+i] = i
        damaged = message[:288] + ecc.interleave(words)
        confidence = [200]*288 + list(ecc.interleave(scores))
        reader = MessageReader(ROBUST80)
        for start in range(0, len(damaged), 17):
            result = reader.feed(damaged[start:start+17], confidence=confidence[start:start+17])
        self.assertEqual(result.payload, payload)
        self.assertEqual(result.filename, 'payload')
        self.assertEqual(result.body_erasures, (6, 0, 0, 0, 0, 0, 0, 0, 6, 0))


if __name__ == '__main__':
    unittest.main()
