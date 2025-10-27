import React, { useEffect, useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import { getCSExamplePlot3, getCSExamplePlot3KMean, getCSExamplePlot3Nemo } from "../../../../../redux/visualizationExampleSlice.jsx";
import ExamplePlot3 from "../plot/ExamplePlot3.jsx";
import { FaProjectDiagram, FaSyncAlt } from "react-icons/fa";
import { downloadPNG, downloadSVG } from "../../../../../shared/utils/downloadPlot.jsx";

const ExampleFigure3 = () => {
    const dispatch = useDispatch();
    const plots = useSelector((s) => s.visualizationExample.plots);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        setLoading(true);
        Promise.all([
            dispatch(getCSExamplePlot3({ option: "corrected" })).unwrap(),
            dispatch(getCSExamplePlot3({ option: "uncorrected" })).unwrap(),
            dispatch(getCSExamplePlot3KMean()).unwrap(),
            dispatch(getCSExamplePlot3Nemo()).unwrap(),
        ])
            .catch(error => console.error("Error fetching Plot3 data:", error))
            .finally(() => setLoading(false));
    }, [dispatch]);

    const renderPlotBox = (label, datasetKey, isError = false) => {
        const ready = plots[`cs_plot3_${datasetKey}`];
        return (
            <div className="bg-base-100 p-4 rounded-lg shadow border border-base-300 flex flex-col">
                <div className="flex items-center justify-between mb-2">
                    <h4 className={`text-md font-semibold ${isError ? 'text-error' : 'text-success'}`}>
                        {label}
                    </h4>
                    {ready && (
                        <div className="flex gap-2">
                            <button
                                className="btn btn-xs btn-outline"
                                onClick={() => downloadSVG(`cs-example-plot3-${datasetKey}`, `plot3-${datasetKey}.svg`)}>
                                SVG
                            </button>
                            <button
                                className="btn btn-xs btn-outline"
                                onClick={() => downloadPNG(`cs-example-plot3-${datasetKey}`, `plot3-${datasetKey}.png`)}>
                                PNG
                            </button>
                        </div>
                    )}
                </div>
                {ready && <ExamplePlot3 dataset={datasetKey} svgId={`cs-example-plot3-${datasetKey}`} />}
            </div>
        );
    };

    return (
        <div className="bg-base-200 p-5 rounded-lg shadow-md border border-base-300 mt-5">
            <h3 className="text-lg font-semibold text-base-content flex items-center gap-2 mb-4">
                <FaProjectDiagram className="text-primary" />
                UMAP visualization of the datasets
            </h3>

            <p className="text-sm text-gray-500">
            This analysis compared the uncorrected dataset with the features extracted from CancerSubminer, colored by the identified cancer subtypes and shaped by the batch. We also visualizes UMAP plots based on the subtyping results from K-means clustering and NEMO.
            </p>

            {loading ? (
                <div className="flex justify-center items-center mt-4 text-primary">
                    <FaSyncAlt className="animate-spin text-xl" />
                    <span className="ml-2">Loading UMAP data...</span>
                </div>
            ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-4">
                    {renderPlotBox("Original Dataset", "uncorrected", true)}
                    {renderPlotBox("CancerSubminer", "corrected", false)}
                    {renderPlotBox("K-Means Clustering", "kmean", true)}
                    {renderPlotBox("NEMO", "nemo", true)}
                </div>
            )}
        </div>
    );
};

export default ExampleFigure3;
