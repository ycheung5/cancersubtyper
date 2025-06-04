from typing import Dict, List, Optional

from pydantic import BaseModel

class HeatmapPlotResponse(BaseModel):
    rowLabel: str
    colLabel: str
    valueLabel: float

class HeatmapPlotTableResponse(BaseModel):
    cluster: Optional[str] = None
    chr: Optional[str] = None
    position: Optional[str] = None
    strand: Optional[str] = None
    ucsc: Optional[str] = None
    genome: Optional[str] = None
    cpg: Optional[str] = None

class DistributionBoxPlotOptionResponse(BaseModel):
    batches: List[str]
    cpg_groups: List[str]

class DistributionBoxPlotResponse(BaseModel):
    batch: str
    subtype: str
    value: float

class UMAPPlotResponse(BaseModel):
    sample_id: str
    x: float
    y: float
    batch: str
    subtype: str

class KMPlotResponse(BaseModel):
    sample_id: str
    os_time: float
    os_event: int
    subtype: str

class KMPlotResult(BaseModel):
    data: List[KMPlotResponse]
    p_value: Optional[float]

class BCtypeFinderComparisonTableResponse(BaseModel):
    sample_id: str
    batch: str
    bctypefinder: Optional[str] = None
    svm: Optional[str] = None
    rf: Optional[str] = None
    lr: Optional[str] = None

class CancerSubminerKMeanPlotResponse(BaseModel):
    sample_id: str
    subtype: str
    x: float
    y: float
    batch: str

class CancerSubminerNemoPlotResponse(BaseModel):
    sample_id: str
    subtype: str
    x: float
    y: float
    batch: str

class CancerSubminerComparisonTableResponse(BaseModel):
    sample_id: str
    batch: str
    cancersubminer: Optional[str] = None
    kmean: Optional[str] = None
    nemo: Optional[str] = None