# Rooftop Segmentation Project Summary

This document outlines the methodology, challenges, and results of the zero-shot rooftop segmentation project targeting three Indian AOIs (Mumbai, Pune, and Vizag).

## Methodology: Zero-Shot Segmentation

Rather than training a semantic segmentation model (like UNet or Mask R-CNN) from scratch on a custom labeled dataset, this project tests the zero-shot capabilities of Meta's **Segment Anything Model (SAM)**, applied via the `segment-geospatial` wrapper.

### Pipeline Overview
1. **Imagery Acquisition:** High-resolution basemap tiles (Esri World Imagery, ~0.3m–0.6m per pixel) are fetched dynamically using `leafmap` bounded by the provided GeoJSON coordinates.
2. **Segmentation (SAM):** The `vit_h` (Huge) model is run in automatic grid-point mode (`points_per_side=32`), generating unclassified masks across the entire image.
3. **Filtering & Post-Processing:**
   - **Area constraints:** Filtered to 15m² – 3000m² to remove cars, noise, and massive fields.
   - **Rectangularity:** A threshold of `>= 0.6` (area / minimum bounding rectangle area) successfully eliminates winding roads and erratic shapes.
   - **Vegetation Filter:** A Green-Ratio index (`g / (r+g+b)`) is calculated, dropping any polygon where a significant fraction of pixels are classified as vegetation.

### Experimental Approach: Point-Prompted SAM
In a secondary experimental notebook (`Roof (1).ipynb`), an alternative approach was tested: instead of generating grids automatically, known building centroids (extracted from Microsoft's Building Footprints) were passed to SAM as exact positive prompts. 
- **Result:** This approach yields significantly higher confidence scores for individual buildings (~0.7 to 0.88 vs ~0.56 in automatic mode) and reduces false positives in empty areas, but it relies on having prior footprint data.

---

## Results & Quantitative Evaluation

Using **Microsoft Global ML Building Footprints** as an independent ground truth, I computed Intersection over Union (IoU), Precision, Recall, and F1-Score for the SAM zero-shot predictions. The vegetation filter successfully improved precision with minimal impact on recall.

**Pixel-Level Segmentation Performance (After Vegetation Filter):**
| AOI | SAM Detections | MS Ground Truth | IoU | Precision | Recall | F1-Score |
|---|---|---|---|---|---|---|
| **Mumbai** | 507 | 560 | 0.1022 | 0.2867 | 0.1370 | 0.1854 |
| **Pune** | 1,572 | 2,282 | 0.2823 | 0.4236 | 0.4584 | 0.4403 |
| **Vizag** | 486 | 630 | 0.3269 | 0.4653 | 0.5237 | 0.4928 |

**Instance-Level Matching (IoU ≥ 0.5):**
| AOI | True Positives | False Positives | False Negatives | Precision | Recall | F1-Score |
|---|---|---|---|---|---|---|
| **Mumbai** | 19 | 538 | 541 | 0.0341 | 0.0339 | 0.0340 |
| **Pune** | 194 | 1,466 | 2,088 | 0.1169 | 0.0850 | 0.0984 |
| **Vizag** | 134 | 353 | 496 | 0.2752 | 0.2127 | 0.2399 |

### Observations on Performance
- **Vizag** and **Pune** achieved respectable zero-shot pixel IoUs of ~0.28–0.32. The layout is relatively structured, allowing SAM to capture complete roofs.
- **Mumbai** performed poorly (IoU 0.10). The extreme density of highly irregular, overlapping high-rises and tightly packed slums caused SAM to massively over-merge structures. When buildings touch without a clear visual boundary (shadow or road), SAM treats the entire block as one blob. This hurts both pixel-level and instance-level accuracy.

---

## Key Challenges & Limitations

1. **Hardware Constraints:** Running `vit_h` requires significant GPU VRAM. Batching (1000x1000 patches) and aggressive garbage collection (`gc.collect()`, `torch.cuda.empty_cache()`) were required to prevent Colab from crashing.
2. **Dense Area Merging:** Without explicit building-boundary training, SAM struggles to separate adjoined buildings (a common issue in Indian urban planning).
3. **Shadows & Occlusions:** Tree canopy occlusions and heavy building shadows confuse the automatic grid generator, leading to partial roof segments.

## Future Improvements

To take this beyond zero-shot performance:
1. **Fine-Tuning:** Fine-tune SAM's mask decoder specifically on Indian rooftop datasets (e.g., SpaceNet) to teach it the visual boundaries of adjoined structures.
2. **Multi-Spectral Data:** Incorporating NIR (Near-Infrared) bands would make vegetation filtering perfect (via NDVI) rather than relying on a visible green-ratio estimate.
3. **Hybrid Architectures:** Combine point-prompted SAM with traditional UNet segmentation models to generate building proposals first, then use SAM to refine the exact polygon boundaries.
