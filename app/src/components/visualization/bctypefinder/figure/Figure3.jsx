import React, { useEffect, useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import { getBCPlot3 } from "../../../../redux/visualizationSlice.jsx";
import Plot3 from "../plot/Plot3.jsx";
import { FaProjectDiagram, FaSyncAlt } from "react-icons/fa";
import {downloadPNG, downloadSVG} from "../../../../shared/utils/downloadPlot.jsx";

const Figure3 = ({ id }) => {
    const dispatch = useDispatch();
    const plots = useSelector(state => state.visualization.plots);

    const [loading, setLoading] = useState(false);

    useEffect(() => {
        setLoading(true);
        Promise.all([
            dispatch(getBCPlot3({ job_id: id, option: "corrected" })).unwrap(),
            dispatch(getBCPlot3({ job_id: id, option: "uncorrected" })).unwrap()
        ])
            .catch(error => console.error("Error fetching Plot3 data:", error))
            .finally(() => setLoading(false));
    }, [dispatch, id]);

    const renderPlotBox = (label, dataset, group, isError) => {
        const key = `bc_plot3_${dataset}`;
        const ready = plots[key];

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
                                onClick={() => downloadSVG(`bc-plot3-${dataset}-${group}`, `plot3-${dataset}-${group}.svg`)}>
                                SVG
                            </button>
                            <button
                                className="btn btn-xs btn-outline"
                                onClick={() => downloadPNG(`bc-plot3-${dataset}-${group}`, `plot3-${dataset}-${group}.png`)}>
                                PNG
                            </button>
                        </div>
                    )}
                </div>
                {ready && <Plot3 dataset={dataset} group={group} svgId={`bc-plot3-${dataset}-${group}`} />}
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
                This section displays UMAP projections of CpG clusters to illustrate both
                <strong> batch effects</strong> and <strong> subtype distributions</strong>.
                The corrected dataset reflects data after batch effect adjustment by the BCtypeFinder model.
            </p>

            {loading && (
                <div className="flex justify-center items-center mt-4 text-primary">
                    <FaSyncAlt className="animate-spin text-xl" />
                    <span className="ml-2">Loading UMAP data...</span>
                </div>
            )}

            {!loading && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-4">
                    {renderPlotBox("Uncorrected Dataset (Batch)", "uncorrected", "batch", true)}
                    {renderPlotBox("Uncorrected Dataset (Subtype)", "uncorrected", "subtype", true)}
                    {renderPlotBox("BCtypeFinder (Batch)", "corrected", "batch", false)}
                    {renderPlotBox("BCtypeFinder (Subtype)", "corrected", "subtype", false)}
                </div>
            )}
        </div>
    );
};

export default Figure3;
