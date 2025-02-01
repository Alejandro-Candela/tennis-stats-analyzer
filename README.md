
# Tennis Stats Analyzer

## Introduction
This project analyzes Tennis players in a video to measure their speed, ball shot speed and number of shots. This project will detect players and the tennis ball using YOLO and also utilizes CNNs to extract court keypoints. This hands on project is perfect for polishing your machine learning, and computer vision skills. 

## Output Videos
Here is a screenshot from one of the output videos:

![Screenshot](output_videos/screenshot.jpeg)

## Models Used
* YOLO v8 for player detection
* Fine Tuned YOLO for tennis ball detection
* Court Key point extraction


## Training
* Tennis ball detetcor with YOLO: training/tennis_ball_detector_training.ipynb
* Tennis court keypoint with Pytorch: training/tennis_court_keypoints_training.ipynb

## Requirements
* python3.8
* ultralytics
* pytroch
* pandas
* numpy 
* opencv

```mermaid
graph TB
    User((User))

    subgraph "Tennis Stats Analyzer System"
        subgraph "Video Processing Container"
            VideoReader["Video Reader<br>OpenCV"]
            VideoWriter["Video Writer<br>OpenCV"]
            VideoUtils["Video Utils<br>Python"]
        end

        subgraph "Object Detection Container"
            subgraph "Player Detection"
                PlayerTracker["Player Tracker<br>YOLO v8"]
                PlayerDetector["Player Detector<br>YOLO v8x"]
            end

            subgraph "Ball Detection"
                BallTracker["Ball Tracker<br>YOLO v5"]
                BallInterpolator["Ball Interpolator<br>Pandas"]
                BallShotDetector["Ball Shot Detector<br>Python"]
            end
        end

        subgraph "Court Analysis Container"
            CourtLineDetector["Court Line Detector<br>ResNet50"]
            KeypointExtractor["Keypoint Extractor<br>PyTorch"]
        end

        subgraph "Statistics Container"
            MiniCourt["Mini Court<br>Python"]
            StatsCalculator["Stats Calculator<br>Python"]
            StatsDrawer["Stats Drawer<br>OpenCV"]
        end

        subgraph "Utility Container"
            BBoxUtils["Bbox Utils<br>Python"]
            Conversions["Conversions<br>Python"]
            Constants["Constants<br>Python"]
        end
    end

    User -->|"Provides video"| VideoReader
    VideoReader -->|"Frames"| PlayerTracker
    VideoReader -->|"Frames"| BallTracker
    VideoReader -->|"First Frame"| CourtLineDetector

    PlayerTracker -->|"Uses"| PlayerDetector
    BallTracker -->|"Uses"| BallInterpolator
    BallTracker -->|"Uses"| BallShotDetector

    CourtLineDetector -->|"Uses"| KeypointExtractor

    PlayerTracker -->|"Player positions"| MiniCourt
    BallTracker -->|"Ball positions"| MiniCourt
    CourtLineDetector -->|"Court keypoints"| MiniCourt

    MiniCourt -->|"Court data"| StatsCalculator
    StatsCalculator -->|"Stats"| StatsDrawer

    PlayerTracker -->|"Uses"| BBoxUtils
    BallTracker -->|"Uses"| BBoxUtils
    MiniCourt -->|"Uses"| Conversions
    StatsCalculator -->|"Uses"| Constants

    StatsDrawer -->|"Annotated frames"| VideoWriter
    VideoWriter -->|"Output video"| User
```