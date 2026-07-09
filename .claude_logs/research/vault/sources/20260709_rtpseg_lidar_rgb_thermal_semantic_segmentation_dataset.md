---
title: RTPSeg: A multi-modality dataset for LiDAR point cloud semantic segmentation assisted with RGB-thermal images in autonomous driving
tags: [source-candidate, weekly-sweep, dataset,_multimodal_semantic_segmentation,_lidar_rgb_thermal,_3d_segmentation]
created: 2026-07-09
source: https://doi.org/10.1016/j.isprsjprs.2026.01.008 ; https://github.com/sssssyf/RTPSeg
status: verified-candidate
---

# RTPSeg: A multi-modality dataset for LiDAR point cloud semantic segmentation assisted with RGB-thermal images in autonomous driving

## Verification

- **Venue/year:** ISPRS Journal of Photogrammetry and Remote Sensing / 2026
- **Primary source(s):** https://doi.org/10.1016/j.isprsjprs.2026.01.008 ; https://github.com/sssssyf/RTPSeg
- **Verification status:** Verified OpenAlex DOI metadata and official GitHub README/API metadata on 2026-07-09.
- **Priority:** high

## Why this matters for [[00_MOC_26_MultimodalSeg]]

New RGB+thermal+LiDAR point-cloud semantic segmentation dataset (3k synchronized frames, 248M point annotations per README); useful for multimodal dense-prediction benchmarks beyond 2D RGB-X.

## Project category

- dataset; multimodal_semantic_segmentation; lidar_rgb_thermal; 3d_segmentation

## Connections

- [[sources/04_weekly_source_sweep_log]]
- [[relatedworks/90_clustered_relatedwork_synthesis]]
- [[PROJECT_TRACKING_26_MultimodalSeg]]

## Next extraction tasks

- Verify full paper PDF/project page details before citing exact numbers.
- Extract benchmark tables, datasets, metrics, and model components if this source becomes part of the final related-work matrix.
- Compare mechanism against RBMA/CGMoD positioning: reliability-aware attention bias, SAM2 memory adaptation, multimodal fusion, or efficient dense prediction as applicable.
