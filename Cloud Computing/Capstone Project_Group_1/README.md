# Capstone Project — Group 1

# Title: 
**FloodWatch: An Automated Cloud-Based Workflow for Multi-Source Flood Monitoring Across CONUS**

By: Ehsan Kahrizi, Arman Oliazadeh, Sunil Bista

**Abstract**

Federal flood information in the United States (U.S.) is distributed across multiple agency systems, each serving a different purpose and following a different update cycle, which makes it difficult for emergency managers and public to form a single, current picture of flood conditions. The National Weather Service’s (NWS) National Water Prediction Service (NWPS) integrates real-time gage observations, short-term forecasts, and modeled inundation extents, while the Federal Emergency Management Agency’s National Flood Hazard Layer (NFHL) separately maintains long-term regulatory flood-hazard zones. Comparing the two currently requires consulting two independently maintained portals. This study presents FloodWatch, an automated, cloud-based workflow that retrieves, standardizes, and jointly visualizes flood-related data from four federal sources: NWS Flood Inundation Mapping, active NWS flood alerts, FEMA's NFHL, and NWPS gage observations across the contiguous United States (CONUS). The platform is deployed as a containerized application on a Jetstream cloud instance, automatically refreshing each data source on an hourly cycle and converting retrieved records into a common GeoJSON representation while preserving each source's provenance, purpose, and update timing. Rather than merging sources into a single inferred flood boundary, the platform is designed to support direct comparison among them, since no single source represents complete or current ground truth. Results are presented for refreshing reliability, data latency, processing time, and spatial agreement between polygon sources using an intersection-over-union metric, along with functional evaluation of the web-mapping interface. The contribution of this study is a reproducible, automatically updating cloud pipeline that brings together previously siloed federal flood information products into a single, source-transparent, national-scale view.


# Outcomes:

* The Docker image is publicly available in this Docker repository: https://hub.docker.com/r/ehsankahrizi1991/floodops-cloud.
* The real-time products are available on these URLs: http://149.165.170.6:8000/ (interactive flood map)
* Flood polygon GeoJSON: http://149.165.170.6:8000/flood_map.geojson
* Data source summary: http://149.165.170.6:8000/sources
* Evaluation metrics: http://149.165.170.6:8000/evaluation
