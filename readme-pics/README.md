# README images

## Delta comparison — DrivingPOV

`delta-comparison-drivingpov.png` is an unchanged copy of the visual comparison
produced during the local embedding-strength experiment. It shows decoded frame
75 (zero-based, approximately 2.5 seconds) from the prepared 720p30 cover and the
lossless encoded files at delta80, delta100, and delta120. All variants carry the
same payload. These are pre-YouTube images, not frames from the viewing proxies
or platform downloads. The contact sheet was rendered at 2560×1600; open it at
full size to compare localized chroma artifacts. A still does not show flicker.

Source footage: **Netflix, DrivingPOV**, obtained from the
[Xiph test-video catalogue](https://media.xiph.org/video/derf/), file
`Netflix_DrivingPOV_4096x2160_60fps_10bit_420.webm`.
The source was resized/cropped and converted to 720p30 YUV420p for testing;
the three encoded variants additionally modify the U chroma plane.
See [Netflix Open Content](https://opencontent.netflix.com/) for source information.

The original experiment artifacts remain locally ignored under `materials/`.
This directory contains only the selected images and credits used by the public README.
