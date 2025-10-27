import React, { useEffect, useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import { getBCExamplePlot4Table } from "../../../../../redux/visualizationExampleSlice.jsx";
import ExamplePlot4Table from "../plot/ExamplePlot4Table.jsx";
import { FaTable, FaSyncAlt } from "react-icons/fa";

const ExampleFigure4 = () => {
    const dispatch = useDispatch();
    const { bc_plot4_table } = useSelector((s) => s.visualizationExample.plots);

    const [loading, setLoading] = useState(true);

    useEffect(() => {
        setLoading(true);
        dispatch(getBCExamplePlot4Table())
            .unwrap()
            .catch((e) => console.error("Error fetching BCExamplePlot4Table:", e))
            .finally(() => setLoading(false));
    }, [dispatch]);

    return (
        <div className="bg-base-200 p-5 rounded-lg shadow-md border border-base-300 mt-5">
            {/* Section Header */}
            <h4 className="text-lg font-semibold text-base-content flex items-center gap-2 mb-3">
                <FaTable className="text-primary" />
                Cancer subtype prediction results
            </h4>

            {/* Description */}
            <p className="text-sm text-gray-600 mb-3">
            This table provides subtype prediction comparison between BCtypeFinder and baseline machine learning models: Support vector machine (SVM), Random Forest, and Logistic regression. The machine learning models were implemented using Scikit-learn package with default parameters.
            </p>

            {/* Loading Indicator */}
            {loading && (
                <div className="flex justify-center items-center mt-4 text-primary">
                    <FaSyncAlt className="animate-spin text-xl" />
                    <span className="ml-2">Loading classification results...</span>
                </div>
            )}

            {/* Table */}
            {!loading && bc_plot4_table && <ExamplePlot4Table />}
        </div>
    );
};

export default ExampleFigure4;
