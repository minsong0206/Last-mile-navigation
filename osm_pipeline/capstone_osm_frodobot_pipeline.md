# Capstone Design Project Context: FrodoBots Dataset and OSM-based Navigation Map Input

## 1. Project Overview

This project aims to build a navigation model that uses visual observations and map-based route images as input.

The main idea is to use the FrodoBots driving dataset and generate OpenStreetMap-based local route map images that can be used as goal/map inputs for a NoMaD-style navigation model.

Instead of using only a goal image, we want to provide the model with a cropped map image containing the planned route from OpenStreetMap. This map image should represent the local path that the robot should follow from its current position.

## 2. Current Goal

The current goal is to find FrodoBots trajectories whose odometry or GPS-based path is highly similar to the corresponding path on OpenStreetMap.

To do this, we want to compare:

1. The odometry or GPS trajectory from the FrodoBots dataset.
2. The route or path generated from OpenStreetMap using the same start point and end point.

After finding FrodoBots data samples whose actual trajectory and OSM route are similar, we will use those samples to generate model input data.

## 3. Why This Matching Step Is Needed

The FrodoBots dataset contains real-world driving trajectories, but not every trajectory may match well with an OSM-generated route.

If the actual robot trajectory and the OSM route are very different, the cropped OSM map image may give incorrect path information to the model.

Therefore, before generating training data, we need to select only the dataset sequences where:

- The FrodoBots trajectory and the OSM route have similar geometry.
- The start point and end point are consistent.
- The local road shape from OSM matches the actual movement pattern.
- The route image can reasonably represent the path followed by the robot.

## 4. Data Sources

### 4.1 FrodoBots Dataset

The FrodoBots dataset provides driving data such as:

- Front camera images
- GPS coordinates
- Odometry or trajectory information
- Timestamped sensor data

We want to use the GPS and/or odometry information to reconstruct the robot’s actual path.

### 4.2 OpenStreetMap

OpenStreetMap will be used to generate map-based route information.

Using the GPS start point and end point from the FrodoBots trajectory, we want to request or compute an OSM-based route.

The OSM route will then be converted into a local map image.

## 5. Trajectory Matching Plan

For each candidate FrodoBots sequence:

1. Extract the GPS or odometry trajectory from the dataset.
2. Select the start point and end point of the trajectory.
3. Generate an OpenStreetMap route using the same start and end points.
4. Convert both trajectories into a comparable coordinate or image representation.
5. Compare the FrodoBots trajectory and the OSM route.
6. Select only the sequences with high similarity.

## 6. Possible Comparison Methods

Several methods can be used to compare the FrodoBots trajectory and the OSM route.

### 6.1 Image-based Comparison

Convert both paths into binary or RGB trajectory images.

- FrodoBots trajectory image
- OSM route image

Then compare the two images using metrics such as:

- L1 loss
- L2 loss
- IoU between path masks
- Chamfer distance between path pixels
- Structural similarity if needed

This method is useful because the final model input will also be an image.

### 6.2 Coordinate-based Comparison

Compare the two trajectories directly in coordinate space.

Possible metrics:

- Average point-to-curve distance
- Dynamic Time Warping distance
- Hausdorff distance
- Fréchet distance
- Start/end point error
- Heading difference along the path

This method may be more geometrically accurate than simple image comparison.

### 6.3 Combined Score

A final matching score can be computed by combining several factors:

```text
matching_score =
    trajectory_shape_similarity
  + start_end_consistency
  + heading_consistency
  + local_road_geometry_similarity

## Dataset Construction After OSM Trajectory Image Generation

After generating the OpenStreetMap trajectory image, I want to build a training dataset by pairing each OSM trajectory image with the corresponding FrodoBots camera images and GPS information.

For each timestamp `t`, one training sample should contain:

- One current front camera image at time `t`
- Two past context images before time `t`
- One OpenStreetMap trajectory image matched with time `t`
- GPS information corresponding to the current frame
- Optional future trajectory or action labels if available

The purpose of this dataset structure is to make the model learn navigation behavior using both visual observations and map-based route information.

Example dataset structure:

```text
sample_t/
├── current_image.jpg          # Camera image at time t
├── past_image_1.jpg           # Previous context image
├── past_image_2.jpg           # Older context image
├── osm_trajectory_image.png   # OSM route/trajectory crop matched with time t
└── metadata.json              # GPS, timestamp, sequence id, and label information

{
  "sequence_id": "frodobot_xxxx",
  "timestamp": "t",
  "current_gps": {
    "latitude": 0.0,
    "longitude": 0.0
  },
  "past_gps": [
    {
      "latitude": 0.0,
      "longitude": 0.0
    },
    {
      "latitude": 0.0,
      "longitude": 0.0
    }
  ],
  "osm_route_image": "osm_trajectory_image.png",
  "current_image": "current_image.jpg",
  "past_images": [
    "past_image_1.jpg",
    "past_image_2.jpg"
  ]
}

## Additional Episode Filtering Criteria

Before generating OSM trajectory images, the FrodoBots episodes should be filtered more strictly.

The filtering criteria are:

1. Remove stationary frames.
   - Frames where the robot is not moving should be removed.
   - Stationary segments can negatively affect trajectory matching and model training.

2. Remove non-sidewalk driving scenes.
   - Since this project targets sidewalk-based robot navigation, trajectories that do not follow sidewalks or pedestrian paths in OpenStreetMap should be excluded.
   - Segments on roads, parking lots, or irrelevant areas should not be used.

3. Keep only episodes with a clear start and end point.
   - Some FrodoBots episodes may contain multiple disconnected segments.
   - In some cases, a new trajectory starts from a different location within the same episode.
   - These episodes should either be split into clean sub-episodes or removed.
   - Each selected episode should contain one continuous trajectory with a clear start point and end point.

The final selected episodes should be clean, continuous, sidewalk-based trajectories that can be reliably matched with OSM routes and used to generate route-conditioned map images.