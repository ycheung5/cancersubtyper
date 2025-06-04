import json
import os
from http import HTTPStatus

import numpy as np
import pandas as pd
from lifelines.statistics import logrank_test, multivariate_logrank_test
from fastapi import APIRouter, Depends, HTTPException
from multipart import file_path
from sqlalchemy.orm import Session
from collections import OrderedDict
from typing import Dict, List

from database import get_db
from helpers.security import get_current_user
from models import User
from repository.job_repository import JobRepository
from schemas.visualization import (
    HeatmapPlotResponse, DistributionBoxPlotResponse,
    UMAPPlotResponse, BCtypeFinderComparisonTableResponse, KMPlotResponse,
    HeatmapPlotTableResponse, KMPlotResult, DistributionBoxPlotOptionResponse,
    CancerSubminerKMeanPlotResponse, CancerSubminerNemoPlotResponse, CancerSubminerComparisonTableResponse
)
from util.path_untils import job_result_path

router = APIRouter(prefix="/visualization", tags=["visualization"])


def get_result_path(job_id: int, db: Session):
    """Retrieve the result path for a job, ensuring it exists and is completed."""
    job_repo = JobRepository(db)
    job = job_repo.get_job_by_id(job_id)
    if not job:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="Job not found")
    if job.status != "Completed":
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="Job status is not completed")
    return job_result_path(job.user_id, job.project_id, job.id)

def load_csv_or_raise(file_path: str, index: bool = True) -> pd.DataFrame:
    """Load a CSV file, raising an HTTPException if not found."""
    if not os.path.exists(file_path):
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="File not found")
    if index:
        return pd.read_csv(file_path, index_col=0)
    else:
        return pd.read_csv(file_path)

def correlation_heatmap_options(job_id: int, db: Session):
    """Retrieve batch-subtype mapping for visualization options."""
    file_path = os.path.join(get_result_path(job_id, db), "preprocessed_dataset.csv")
    all_sample_df = load_csv_or_raise(file_path)

    if "Batch" not in all_sample_df.columns or "subtype" not in all_sample_df.columns:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="Required columns not found in dataset.")

    batch_subtypes: Dict[str, List[str]] = all_sample_df.groupby('Batch')['subtype'].apply(
        lambda x: list(set(x))).to_dict()
    all_subtypes = list(set(all_sample_df["subtype"]))

    ordered_batch_subtypes = OrderedDict([("All", all_subtypes)] + list(batch_subtypes.items()))
    return ordered_batch_subtypes

def correlation_heatmap(job_id: int, batch: str, subtype: str, db: Session):
    """Retrieve heatmap data for a given batch and subtype."""
    file_path = os.path.join(get_result_path(job_id, db), f"{batch}_{subtype}_heatmap.csv")
    heatmap_df = load_csv_or_raise(file_path)

    heatmap_long_df = heatmap_df.melt(ignore_index=False, var_name="colLabel", value_name="valueLabel")
    heatmap_long_df.reset_index(inplace=True)
    heatmap_long_df.rename(columns={"index": "rowLabel"}, inplace=True)

    try:
        return [
            HeatmapPlotResponse(
                rowLabel=f"CpG Cluster {row}",
                colLabel=f"CpG Cluster {col}",
                valueLabel=float(val)
            )
            for row, col, val in heatmap_long_df.itertuples(index=False)
        ]
    except ValueError:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="No data found")

def correlation_heatmap_table(job_id: int, clusters: str, db: Session):
    """Retrieve CpG information table, filtering by clusters."""
    file_path = os.path.join(get_result_path(job_id, db), "cpg_info_table.csv")
    cpg_info_df = load_csv_or_raise(file_path, False)

    if "cluster" not in cpg_info_df.columns:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="Cluster column missing in dataset.")

    cluster_list = [c.strip() for c in clusters.split(",")]
    cpg_info_df["cluster"] = cpg_info_df["cluster"].astype(str)  # Convert cluster to string before filtering
    filtered_df = cpg_info_df[cpg_info_df["cluster"].isin(cluster_list)]

    return [
        HeatmapPlotTableResponse(
            cluster=str(row.cluster),
            cpg=str(row.CpG),
            chr=str(row.CHR),
            position=str(row.Position),
            strand=str(row.Strand),
            ucsc=str(row.UCSC_RefGene_Name) if pd.notna(row.UCSC_RefGene_Name) else None,
            genome=str(row.Genome_Build),
        )
        for row in filtered_df.itertuples(index=False)
    ]

def distribution_boxplot_options(job_id: int, db: Session):
    """Retrieve available CpG cluster options from all_samples_table.csv."""
    file_path = os.path.join(get_result_path(job_id, db), "preprocessed_dataset.csv")
    all_sample_df = load_csv_or_raise(file_path)

    # Remove non-cluster columns before returning options
    cluster_options = list(all_sample_df.columns)
    if "Batch" in cluster_options:
        cluster_options.remove("Batch")
    if "subtype" in cluster_options:
        cluster_options.remove("subtype")

    batches = all_sample_df['Batch'].unique().tolist()

    return DistributionBoxPlotOptionResponse(batches=batches, cpg_groups=cluster_options)

def distribution_boxplot(job_id: int, option: str, batch: str, db: Session):
    """Retrieve values for a selected CpG cluster."""
    file_path = os.path.join(get_result_path(job_id, db), "preprocessed_dataset.csv")
    all_sample_df = load_csv_or_raise(file_path)

    if option not in all_sample_df.columns:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=f"CpG Cluster '{option}' not found in dataset.")

    if batch != "All":
        all_sample_df = all_sample_df[all_sample_df["Batch"] == batch]

    return [
        DistributionBoxPlotResponse(
            batch=row["Batch"],
            subtype=row["subtype"],
            value=row[option]
        )
        for index, row in all_sample_df.iterrows()
    ]

def umap_plot(job_id: int, option: str, db: Session, filter_source: bool = False):
    """Retrieve UMAP embedding for corrected or uncorrected data."""
    if option not in ["corrected", "uncorrected"]:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=f"Invalid option '{option}'")

    file_path = os.path.join(get_result_path(job_id, db), f"{option}_umap_embedding.csv")
    umap_df = load_csv_or_raise(file_path)

    if filter_source:
        umap_df = umap_df[umap_df["Batch"] == "Source"]

    return [
        UMAPPlotResponse(
            sample_id=str(index),
            x=row.x,
            y=row.y,
            batch=row.Batch,
            subtype=row.subtype,
        )
        for index, row in umap_df.iterrows()
    ]

def km_plot_options(job_id: int, db: Session):
    """Retrieve available batch options from km_plot_data.csv."""
    file_path = os.path.join(get_result_path(job_id, db), "km_plot_data.csv")
    km_plot_df = load_csv_or_raise(file_path)

    batches = km_plot_df['Batch'].unique().tolist()

    return batches

def km_plot(job_id: int, batch: str, db: Session):
    """Retrieve KM plot data and compute p-value."""
    file_path = os.path.join(get_result_path(job_id, db), "km_plot_data.csv")
    km_plot_df = load_csv_or_raise(file_path)
    km_plot_df = km_plot_df[km_plot_df["Batch"] == batch]

    # Convert OS_time and OS_event to numeric
    km_plot_df["OS_time"] = pd.to_numeric(km_plot_df["OS_time"], errors="coerce")
    km_plot_df["OS_event"] = pd.to_numeric(km_plot_df["OS_event"], errors="coerce")

    # Ensure no missing values
    km_plot_df.dropna(subset=["OS_time", "OS_event"], inplace=True)

    # Extract unique subtypes and assign numeric labels
    subtypes = km_plot_df["Label_subtype"].unique()
    subtype_mapping = {subtype: idx for idx, subtype in enumerate(subtypes)}

    # Convert categorical subtypes to numeric
    km_plot_df["Subtype_Label"] = km_plot_df["Label_subtype"].map(subtype_mapping)

    # Compute log-rank test (p-value)
    survival_times = km_plot_df["OS_time"].values
    event_observed = km_plot_df["OS_event"].values
    group_labels = km_plot_df["Subtype_Label"].values

    if len(subtypes) >= 2:
        results = multivariate_logrank_test(survival_times, group_labels, event_observed)
        p_value = float(results.p_value) if not np.isnan(results.p_value) else 1.0  # Ensure valid number
    else:
        p_value = None

    return KMPlotResult(
        data=[
            KMPlotResponse(
                sample_id=str(index),
                os_time=row["OS_time"],
                os_event=row["OS_event"],
                subtype=row["Label_subtype"]
            )
            for index, row in km_plot_df.iterrows()
        ],
        p_value=p_value  # Ensure valid numeric output
    )

@router.get("/{job_id}/bctypefinder/plot1-options", status_code=HTTPStatus.OK,
            response_model=Dict[str, List[str]])
async def get_bctypefinder_plot1_options(job_id: int, db: Session = Depends(get_db),
                                         current_user: User = Depends(get_current_user)):
    return correlation_heatmap_options(job_id, db)


@router.get("/{job_id}/bctypefinder/plot1/{batch}/{subtype}", status_code=HTTPStatus.OK,
            response_model=List[HeatmapPlotResponse])
async def get_bctypefinder_plot1(job_id: int, batch: str, subtype: str, db: Session = Depends(get_db),
                                 current_user: User = Depends(get_current_user)):
    return correlation_heatmap(job_id, batch, subtype, db)


@router.get("/{job_id}/bctypefinder/plot1-table/{clusters}", status_code=HTTPStatus.OK,
            response_model=List[HeatmapPlotTableResponse])
async def get_bctypefinder_plot1_table(job_id: int, clusters: str, db: Session = Depends(get_db),
                                       current_user: User = Depends(get_current_user)):
    return correlation_heatmap_table(job_id, clusters, db)

@router.get("/{job_id}/bctypefinder/plot2-options", status_code=HTTPStatus.OK, response_model=DistributionBoxPlotOptionResponse)
async def get_bctypefinder_plot2_options(job_id: int, db: Session = Depends(get_db),
                                         current_user: User = Depends(get_current_user)):
    return distribution_boxplot_options(job_id, db)

@router.get("/{job_id}/bctypefinder/plot2/{option}/{batch}", status_code=HTTPStatus.OK, response_model=List[DistributionBoxPlotResponse])
async def get_bctypefinder_plot2(job_id: int, option: str, batch: str, db: Session = Depends(get_db),
                                 current_user: User = Depends(get_current_user)):
    return distribution_boxplot(job_id, option, batch, db)

@router.get("/{job_id}/bctypefinder/plot3/{option}", status_code=HTTPStatus.OK,
            response_model=List[UMAPPlotResponse])
async def get_bctypefinder_plot3(job_id: int, option: str, db: Session = Depends(get_db),):
                                 # current_user: User = Depends(get_current_user)):
    return umap_plot(job_id, option, db)

@router.get("/{job_id}/bctypefinder/plot4-table", status_code=HTTPStatus.OK,
            response_model=List[BCtypeFinderComparisonTableResponse])
async def get_bctypefinder_plot4(job_id: int, db: Session = Depends(get_db),
                                 current_user: User = Depends(get_current_user)):
    """Retrieve BCtypeFinder classification comparison table."""
    file_path = os.path.join(get_result_path(job_id, db), "result_comparison_ml.csv")
    comparison_table_df = load_csv_or_raise(file_path)

    return [
        BCtypeFinderComparisonTableResponse(
            sample_id=str(index),
            batch=row["Batch"],
            bctypefinder=row["BCtypeFinder"] if pd.notna(row["BCtypeFinder"]) else None,
            svm=row["SVM"] if pd.notna(row["SVM"]) else None,
            rf=row["RF"] if pd.notna(row["RF"]) else None,
            lr=row["LogReg"] if pd.notna(row["LogReg"]) else None
        )
        for index, row in comparison_table_df.iterrows()
    ]

@router.get("/{job_id}/bctypefinder/plot5-options", status_code=HTTPStatus.OK, response_model=List[str])
async def get_bctypefinder_plot5_options(job_id: int, db: Session = Depends(get_db),
                                         current_user: User = Depends(get_current_user)):
    return km_plot_options(job_id, db)

@router.get("/{job_id}/bctypefinder/plot5/{batch}", status_code=HTTPStatus.OK, response_model=KMPlotResult)
async def get_bctypefinder_plot5(job_id: int, batch: str, db: Session = Depends(get_db),
                                 current_user: User = Depends(get_current_user)):
    return km_plot(job_id, batch, db)

@router.get("/{job_id}/cancersubminer/plot1-options", status_code=HTTPStatus.OK,
            response_model=Dict[str, List[str]])
async def get_cancersubminer_plot1_options(job_id: int, db: Session = Depends(get_db),
                                         current_user: User = Depends(get_current_user)):
    return correlation_heatmap_options(job_id, db)

@router.get("/{job_id}/cancersubminer/plot1/{batch}/{subtype}", status_code=HTTPStatus.OK,
            response_model=List[HeatmapPlotResponse])
async def get_cancersubminer_plot1(job_id: int, batch: str, subtype: str, db: Session = Depends(get_db),
                                 current_user: User = Depends(get_current_user)):
    return correlation_heatmap(job_id, batch, subtype, db)

@router.get("/{job_id}/cancersubminer/plot1-table/{clusters}", status_code=HTTPStatus.OK,
            response_model=List[HeatmapPlotTableResponse])
async def get_cancersubminer_plot1_table(job_id: int, clusters: str, db: Session = Depends(get_db),
                                       current_user: User = Depends(get_current_user)):
    return correlation_heatmap_table(job_id, clusters, db)

@router.get("/{job_id}/cancersubminer/plot2-options", status_code=HTTPStatus.OK, response_model=DistributionBoxPlotOptionResponse)
async def get_cancersubminer_plot2_options(job_id: int, db: Session = Depends(get_db),
                                         current_user: User = Depends(get_current_user)):
    return distribution_boxplot_options(job_id, db)

@router.get("/{job_id}/cancersubminer/plot2/{option}/{batch}", status_code=HTTPStatus.OK, response_model=List[DistributionBoxPlotResponse])
async def get_cancersubminer_plot2(job_id: int, option: str, batch: str, db: Session = Depends(get_db),
                                 current_user: User = Depends(get_current_user)):
    return distribution_boxplot(job_id, option, batch, db)

@router.get("/{job_id}/cancersubminer/plot3/{option}", status_code=HTTPStatus.OK,
            response_model=List[UMAPPlotResponse])
async def get_cancersubminer_plot3(job_id: int, option: str, db: Session = Depends(get_db),
                                 current_user: User = Depends(get_current_user)):
    return umap_plot(job_id, option, db, filter_source=(option == "uncorrected"))

@router.get("/{job_id}/cancersubminer/plot3-kmean", status_code=HTTPStatus.OK, response_model=List[CancerSubminerKMeanPlotResponse])
async def get_cancersubminer_plot3_kmean(job_id: int, db: Session = Depends(get_db),
                                         current_user: User = Depends(get_current_user)):
    """Retrieve UMAP coordinates with K-Means cluster labels."""
    result_dir = get_result_path(job_id, db)

    kmeans_csv_path = os.path.join(result_dir, "kmeans.csv")
    umap_csv_path = os.path.join(result_dir, "uncorrected_umap_embedding.csv")

    kmeans_df = pd.read_csv(kmeans_csv_path, index_col=0)
    umap_df = pd.read_csv(umap_csv_path, index_col=0)
    umap_df = umap_df.drop(columns=["subtype"])

    # Join on SampleID index
    merged_df = umap_df.join(kmeans_df)

    # Rename column for consistency
    merged_df.rename(columns={"Cluster": "subtype"}, inplace=True)

    return [
        CancerSubminerKMeanPlotResponse(
            sample_id=str(idx),
            x=row.x,
            y=row.y,
            batch=row.Batch,
            subtype=row.subtype
        )
        for idx, row in merged_df.iterrows()
    ]

@router.get("/{job_id}/cancersubminer/plot3-nemo", status_code=HTTPStatus.OK, response_model=List[CancerSubminerNemoPlotResponse])
async def get_cancersubminer_plot3_nemo(job_id: int, db: Session = Depends(get_db),
                                        current_user: User = Depends(get_current_user)):
    """Retrieve UMAP coordinates with NEMO clustering labels."""
    result_dir = get_result_path(job_id, db)

    nemo_csv_path = os.path.join(result_dir, "nemo.csv")
    umap_csv_path = os.path.join(result_dir, "uncorrected_umap_embedding.csv")

    nemo_df = pd.read_csv(nemo_csv_path, index_col=0)
    umap_df = pd.read_csv(umap_csv_path, index_col=0)
    umap_df = umap_df.drop(columns=["subtype"])

    merged_df = umap_df.join(nemo_df)

    merged_df.rename(columns={"Subtype": "subtype"}, inplace=True)

    return [
        CancerSubminerKMeanPlotResponse(
            sample_id=str(idx),
            x=row.x,
            y=row.y,
            batch=row.Batch,
            subtype=row.subtype
        )
        for idx, row in merged_df.iterrows()
    ]

@router.get("/{job_id}/cancersubminer/plot4-table", status_code=HTTPStatus.OK, response_model=List[CancerSubminerComparisonTableResponse])
async def get_cancersubminer_plot4_table(job_id: int, db: Session = Depends(get_db),
                                         current_user: User = Depends(get_current_user)):
    """Retrieve CancerSubminer classification comparison table."""
    file_path = os.path.join(get_result_path(job_id, db), "result_comparison_ml.csv")
    comparison_table_df = load_csv_or_raise(file_path)

    return [
        CancerSubminerComparisonTableResponse(
            sample_id=str(index),
            batch=row["Batch"],
            cancersubminer=row["CancerSubminer"],
            kmean=row["KMeans"],
            nemo=row["NeMo"]
        )
        for index, row in comparison_table_df.iterrows()
    ]

@router.get("/{job_id}/cancersubminer/plot5-options", status_code=HTTPStatus.OK, response_model=List[str])
async def get_cancersubminer_plot5_options(job_id: int, db: Session = Depends(get_db),
                                         current_user: User = Depends(get_current_user)):
    return km_plot_options(job_id, db)

@router.get("/{job_id}/cancersubminer/plot5/{batch}", status_code=HTTPStatus.OK, response_model=KMPlotResult)
async def get_cancersubminer_plot5(job_id: int, batch: str, db: Session = Depends(get_db),
                                 current_user: User = Depends(get_current_user)):
    return km_plot(job_id, batch, db)