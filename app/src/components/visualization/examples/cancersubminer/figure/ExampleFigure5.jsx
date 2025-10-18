import React, { useEffect, useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import {
    getCSExamplePlot5Option,
    getCSExamplePlot5,
} from "../../../../../redux/visualizationExampleSlice.jsx";
import ExamplePlot5 from "../plot/ExamplePlot5.jsx";
import { FaChartLine, FaFilter, FaSyncAlt } from "react-icons/fa";
import { downloadPNG, downloadSVG } from "../../../../../shared/utils/downloadPlot.jsx";

const ExampleFigure5 = () => {
    const dispatch = useDispatch();
    const { cs_plot5_option, cs_plot5 } = useSelector(
        (s) => s.visualizationExample.plots
    );

    const [loading, setLoading] = useState(true);
    const [selectedBatch, setSelectedBatch] = useState("");

    useEffect(() => {
        setLoading(true);
        dispatch(getCSExamplePlot5Option())
            .unwrap()
            .catch((err) => console.error("Error fetching CSPlot5Option:", err))
            .finally(() => setLoading(false));
    }, [dispatch]);

    useEffect(() => {
        if (cs_plot5_option && cs_plot5_option.length > 0) {
            setSelectedBatch(cs_plot5_option[0]);
        }
    }, [cs_plot5_option]);

    useEffect(() => {
        if (selectedBatch) {
            setLoading(true);
            dispatch(getCSExamplePlot5({ batch: selectedBatch }))
                .unwrap()
                .catch((err) => console.error("Error fetching CSExamplePlot5:", err))
                .finally(() => setLoading(false));
        }
    }, [dispatch, selectedBatch]);

    return (
        <div className="bg-base-200 p-6 rounded-lg shadow-md border border-base-300 mt-5">
            <h3 className="text-xl font-semibold text-base-content flex items-center gap-2 mb-4">
                <FaChartLine className="text-primary" />
                Kaplan-Meier Survival Analysis
            </h3>
            <p className="text-sm text-gray-500 mb-6">
                This Kaplan-Meier plot shows survival probabilities for different subtypes
                across a selected batch. The x-axis indicates time, and the y-axis shows the probability of survival.
            </p>

            <div className="flex flex-wrap gap-4 items-center bg-base-100 p-4 rounded-lg shadow-sm border border-base-300">
                <label className="font-medium text-base-content flex items-center gap-2">
                    <FaFilter className="text-primary" />
                    Batch:
                </label>
                <select
                    className="select select-bordered w-48"
                    value={selectedBatch}
                    onChange={(e) => setSelectedBatch(e.target.value)}
                    disabled={loading}
                >
                    {cs_plot5_option && cs_plot5_option.length > 0 && cs_plot5_option.map((batch) => (
                        <option key={batch} value={batch}>{batch}</option>
                    ))}
                </select>
            </div>

            <div className="mt-4">
                {loading && (
                    <div className="flex justify-center items-center mt-4 text-primary">
                        <FaSyncAlt className="animate-spin text-xl" />
                        <span className="ml-2">Loading Kaplan-Meier plot...</span>
                    </div>
                )}
            </div>

            {!loading && cs_plot5?.data.length > 0 && (
                <div className="bg-base-100 p-6 rounded-lg shadow-md border border-base-300 mt-6">
                    <h4 className="text-lg font-semibold text-base-content flex items-center gap-2 mb-3">
                        <FaChartLine className="text-primary" />
                        Survival Curves by Subtype
                    </h4>
                    <div className="flex justify-end gap-4 mt-4">
                        <button className="btn btn-sm btn-outline" onClick={() => downloadSVG("cs-example-plot5", "plot5.svg")}>
                            Download SVG
                        </button>
                        <button className="btn btn-sm btn-outline" onClick={() => downloadPNG("cs-example-plot5", "plot5.png")}>
                            Download PNG
                        </button>
                    </div>
                    <div className="flex justify-center">
                        <ExamplePlot5 />
                    </div>
                </div>
            )}
        </div>
    );
};

export default ExampleFigure5;
