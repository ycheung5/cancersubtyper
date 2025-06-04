import React, { useEffect, useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import {getCSPlot3, getCSPlot3KMean, getCSPlot3Nemo} from "../../../../redux/visualizationSlice.jsx";
import Plot3 from "../plot/Plot3.jsx";
import { FaProjectDiagram, FaSyncAlt } from "react-icons/fa";
import { downloadPNG, downloadSVG } from "../../../../shared/utils/downloadPlot.jsx";

const Figure3 = ({ id }) => {
    const dispatch = useDispatch();
    const plots = useSelector(state => state.visualization.plots);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        setLoading(true);
        Promise.all([
            dispatch(getCSPlot3({ job_id: id, option: "corrected" })).unwrap(),
            dispatch(getCSPlot3({ job_id: id, option: "uncorrected" })).unwrap(),
            dispatch(getCSPlot3KMean({ job_id: id })).unwrap(),
            dispatch(getCSPlot3Nemo({ job_id: id })).unwrap(),
        ])
            .catch(error => console.error("Error fetching Plot3 data:", error))
            .finally(() => setLoading(false));
    }, [dispatch, id]);

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
                                onClick={() => downloadSVG(`cs-plot3-${datasetKey}`, `plot3-${datasetKey}.svg`)}>
                                SVG
                            </button>
                            <button
                                className="btn btn-xs btn-outline"
                                onClick={() => downloadPNG(`cs-plot3-${datasetKey}`, `plot3-${datasetKey}.png`)}>
                                PNG
                            </button>
                        </div>
                    )}
                </div>
                {ready && <Plot3 dataset={datasetKey} svgId={`cs-plot3-${datasetKey}`} />}
            </div>
        );
    };

    return (
        <div className="bg-base-200 p-5 rounded-lg shadow-md border border-base-300 mt-5">
            <h3 className="text-lg font-semibold text-base-content flex items-center gap-2 mb-4">
                <FaProjectDiagram className="text-primary" />
                UMAP Visualizations
            </h3>

            <p className="text-sm text-gray-500">
                Each point represents a sample, with <strong>color</strong> indicating <strong>subtype</strong> and <strong>shape</strong> indicating <strong>batch</strong>.
                The corrected datasets reflect the removal of batch effects, enabling clearer subtype patterns.
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

export default Figure3;
