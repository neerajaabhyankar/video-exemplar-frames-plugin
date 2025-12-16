## Path to creating a test bed

1. Downloaded a bunch of videos
```
https://www.pexels.com/video/man-playing-basketball-5192069/
https://www.pexels.com/video/low-angle-view-of-a-man-playing-basketball-5192077/
https://www.pexels.com/video/a-man-walking-with-a-basketball-5192149/
https://www.pexels.com/video/a-basketball-player-shooting-the-ball-in-a-reverse-layup-5192154/
https://www.pexels.com/video/spinning-basketball-on-a-man-s-index-finger-5192072/
https://www.pexels.com/video/man-playing-basketball-5192076/
https://www.pexels.com/video/shallow-focus-of-basketball-on-the-ground-5192074/
https://www.pexels.com/video/man-playing-basketball-in-the-morning-5192025/
```
stitched them with pexels_stitch.py

2. Made an IMAGE dataset with its frames (just for removing complexity for now -- will use VIDEO later)<br>
using frames_dataset.py

3. Got their embeddings using embeddings.py

4. Applied zero-shot instance segmentation {person, basketball} to some of the frames

5. Tested Annotation Propagation on some of the frames using quicktests/annoprop_frames.py

6. [replacing 4.] Exported this to https://ha.test.fiftyone.ai/datasets using foe_dataset.py for human annotation