import React, { useEffect, useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import {
    getCSExamplePlot1Option,
    getCSExamplePlot1,
    getCSExamplePlot1Table,
} from "../../../../../redux/visualizationExampleSlice.jsx";
import ExamplePlot1 from "../plot/ExamplePlot1.jsx";
import ExamplePlot1Table from "../plot/ExamplePlot1Table.jsx";
import { FaFilter, FaChartBar, FaTable, FaSyncAlt } from "react-icons/fa";
import { downloadPNG, downloadSVG } from "../../../../../shared/utils/downloadPlot.jsx";

const ExampleFigure1 = () => {
    const dispatch = useDispatch();
    const { cs_plot1_option = {}, cs_plot1, cs_plot1_table } =
        useSelector((state) => state.visualizationExample.plots);

    const [loadingState, setLoadingState] = useState("loading-options");
    const [selectedBatch, setSelectedBatch] = useState("");
    const [selectedSubtype, setSelectedSubtype] = useState("");

    // 1) load options
    useEffect(() => {
        setLoadingState("loading-options");
        dispatch(getCSExamplePlot1Option())
            .unwrap()
            .then(() => setLoadingState("idle"))
            .catch(() => setLoadingState("idle"));
    }, [dispatch]);

    // 2) pick first batch/subtype
    useEffect(() => {
        if (cs_plot1_option && Object.keys(cs_plot1_option).length > 0) {
            const firstBatch = Object.keys(cs_plot1_option).filter((b) => b !== "All")[0] || "";
            const firstSubtype = cs_plot1_option[firstBatch]?.[0] || "";
            setSelectedBatch(firstBatch);
            setSelectedSubtype(firstSubtype);
        }
    }, [cs_plot1_option]);

    // 3) fetch heatmap, then table
    useEffect(() => {
        if (!selectedBatch || !selectedSubtype) return;
        setLoadingState("loading-heatmap");
        dispatch(getCSExamplePlot1({ batch: selectedBatch, subtype: selectedSubtype }))
            .unwrap()
            .then((res) => {
                // infer clusters from heatmap (CpG Cluster {id})
                const clusters = new Set(
                    res
                        ?.map(({ rowLabel, colLabel }) => [
                            String(rowLabel).replace("CpG Cluster ", "").trim(),
                            String(colLabel).replace("CpG Cluster ", "").trim(),
                        ])
                        .flat() ?? []
                );
                const clustersString = [...clusters].join(",");
                setLoadingState("loading-table");
                return dispatch(getCSExamplePlot1Table({ clusters: clustersString })).unwrap();
            })
            .then(() => setLoadingState("idle"))
            .catch(() => setLoadingState("idle"));
    }, [dispatch, selectedBatch, selectedSubtype]);

    const batchOptions = Object.keys(cs_plot1_option).filter((b) => b !== "All");
    const subtypes = selectedBatch ? cs_plot1_option[selectedBatch] ?? [] : [];

    return (
        <div className="bg-base-2 00 p-6 rounded-lg shadow-md border border-base-300">
            {/* Title */}
            <h3 className="text-xl font-semibold text-base-content flex items-center gap-2 mb-4">
                <FaChartBar className="text-primary" />
                CpG Cluster Analysis (Example)
            </h3>
            <p className="text-sm text-gray-500 mb-6">
                Explore CpG clusters across batches/subtypes using bundled example data.
            </p>

            {/* Filters */}
            <div className="flex flex-wrap gap-4 items-center bg-base-100 p-4 rounded-lg shadow-sm border border-base-300">
                <label className="font-medium text-base-content flex items-center gap-2">
                    <FaFilter className="text-primary" />
                    Batch:
                </label>
                <select
                    value={selectedBatch}
                    onChange={(e) => {
                        const b = e.target.value;
                        setSelectedBatch(b);
                        setSelectedSubtype(cs_plot1_option[b]?.[0] || "");
                    }}
                    className="select select-bordered w-48"
                    disabled={loadingState === "loading-options"}
                >
                    {batchOptions.length ? (
                        batchOptions.map((b) => (
                            <option key={b} value={b}>
                                {b}
                            </option>
                        ))
                    ) : (
                        <option value="">No Batches Available</option>
                    )}
                </select>

                <label className="font-medium text-base-content flex items-center gap-2">
                    <FaFilter className="text-primary" />
                    Subtype:
                </label>
                <select
                    value={selectedSubtype}
                    onChange={(e) => setSelectedSubtype(e.target.value)}
                    className="select select-bordered w-48"
                    disabled={loadingState === "loading-options"}
                >
                    {subtypes.length ? (
                        subtypes.map((s) => (
                            <option key={s} value={s}>
                                {s}
                            </option>
                        ))
                    ) : (
                        <option value="">No Subtypes Available</option>
                    )}
                </select>
            </div>

            {/* Loading states */}
            {loadingState === "loading-options" && (
                <div className="flex justify-center items-center mt-4 text-primary">
                    <FaSyncAlt className="animate-spin text-xl" />
                    <span className="ml-2">Loading filter options…</span>
                </div>
            )}
            {loadingState === "loading-heatmap" && (
                <div className="flex justify-center items-center mt-4 text-primary">
                    <FaSyncAlt className="animate-spin text-xl" />
                    <span className="ml-2">Loading heatmap…</span>
                </div>
            )}
            {loadingState === "loading-table" && (
                <div className="flex justify-center items-center mt-4 text-primary">
                    <FaSyncAlt className="animate-spin text-xl" />
                    <span className="ml-2">Loading table…</span>
                </div>
            )}

            {/* Heatmap */}
            {loadingState === "idle" && (cs_plot1?.length ?? 0) > 0 && (
                <div className="bg-base-100 p-6 rounded-lg shadow-md border border-base-300 mt-6">
                    <h4 className="text-lg font-semibold text-base-content flex items-center gap-2 mb-3">
                        <FaChartBar className="text-primary" />
                        Correlation Heatmap (Example)
                    </h4>
                    <p className="text-sm text-gray-500 mb-6">
                        Spearman correlations between CpG clusters for the selected batch/subtype.
                    </p>

                    <div className="flex justify-end gap-4">
                        <button
                            className="btn btn-sm btn-outline"
                            onClick={() => downloadSVG("cs-example-plot1", "example-plot1.svg")}
                        >
                            Download SVG
                        </button>
                        <button
                            className="btn btn-sm btn-outline"
                            onClick={() => downloadPNG("cs-example-plot1", "example-plot1.png")}
                        >
                            Download PNG
                        </button>
                    </div>

                    <div className="flex justify-center mt-4" id="cs-example-plot1">
                        <div className="w-full max-w-4xl">
                            <ExamplePlot1 />
                        </div>
                    </div>
                </div>
            )}

            {/* Table */}
            {loadingState === "idle" && cs_plot1_table && (
                <div className="bg-base-100 p-6 rounded-lg shadow-md border border-base-300 mt-6">
                    <h4 className="text-lg font-semibold text-base-content flex items-center gap-2 mb-3">
                        <FaTable className="text-primary" />
                        CpG Cluster Details (Example)
                    </h4>
                    <p className="text-sm text-gray-500 mb-6">
                        Attributes of CpG clusters shown in the heatmap above.
                    </p>

                    <ExamplePlot1Table />
                </div>
            )}
        </div>
    );
};

export default ExampleFigure1;
