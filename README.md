# CancerSubtyper

## Abstract

**CancerSubtyper** is a web-based platform for deep learning-based cancer subtyping using DNA methylation data. It supports both supervised and semi-supervised workflows for predicting or discovering molecular subtypes. Users can upload methylation datasets (with or without subtype labels), run models, and explore interactive visualizations such as UMAP projections, CpG heatmaps, and Kaplan-Meier survival plots.

The platform currently includes:
- **BCtypeFinder** – a supervised classifier trained on TCGA-BRCA for intrinsic breast cancer subtype prediction.
- **CancerSubminer** – a semi-supervised model that performs subtype discovery or refinement with optional clustering constraints.

This tool is designed to be accessible to non-programmers while remaining robust enough for advanced molecular analysis.

## Requirements

- Docker
- Docker Compose

## Usage

CancerSubtyper includes:
- A **React frontend** for user interaction
- A **FastAPI backend** that manages models, file storage, and asynchronous job execution
- Background workers and databases managed using **Celery**, **Redis**, and **PostgreSQL**

## How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/cancersubtyper.git
cd cancersubtyper
docker compose up --build
```
