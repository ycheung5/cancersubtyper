import React, { useState, useMemo } from "react";
import { useSelector } from "react-redux";
import { FaSearch, FaSort, FaSortUp, FaSortDown } from "react-icons/fa";

const Plot4Table = () => {
    const plots = useSelector(state => state.visualization.plots);
    const data = useMemo(() => plots["bc_plot4_table"] || [], [plots]);

    const [searchQuery, setSearchQuery] = useState("");
    const [searchColumn, setSearchColumn] = useState("all");
    const [sortColumn, setSortColumn] = useState(null);
    const [sortOrder, setSortOrder] = useState("asc");
    const [currentPage, setCurrentPage] = useState(1);
    const rowsPerPage = 10;

    const columns = [
        { key: "sample_id", label: "Sample" },
        { key: "batch", label: "Batch" },
        { key: "bctypefinder", label: "BCTypeFinder" },
        { key: "svm", label: "SVM" },
        { key: "rf", label: "Random Forest" },
        { key: "lr", label: "Logistic Regression" },
    ];

    const filteredData = useMemo(() => {
        return data.filter(row => {
            if (!searchQuery) return true;
            if (searchColumn === "all") {
                return Object.values(row).some(value =>
                    value?.toString().toLowerCase().includes(searchQuery.toLowerCase())
                );
            }
            return row[searchColumn]?.toString().toLowerCase().includes(searchQuery.toLowerCase());
        });
    }, [data, searchQuery, searchColumn]);

    const sortedData = useMemo(() => {
        if (!sortColumn) return filteredData;

        return [...filteredData].sort((a, b) => {
            let valA = a[sortColumn];
            let valB = b[sortColumn];

            if (!isNaN(valA) && !isNaN(valB)) {
                valA = Number(valA);
                valB = Number(valB);
            }

            return sortOrder === "asc" ? (valA > valB ? 1 : -1) : (valA < valB ? 1 : -1);
        });
    }, [filteredData, sortColumn, sortOrder]);

    const totalPages = Math.max(1, Math.ceil(sortedData.length / rowsPerPage));
    const paginatedData = useMemo(() => {
        const startIndex = (currentPage - 1) * rowsPerPage;
        return sortedData.slice(startIndex, startIndex + rowsPerPage);
    }, [sortedData, currentPage, rowsPerPage]);

    const handleSort = (columnKey) => {
        setSortColumn(columnKey);
        setSortOrder(prevOrder => (prevOrder === "asc" ? "desc" : "asc"));
    };

    return (
        <div className="p-6 bg-base-100 rounded-lg shadow-md border border-base-300">
            {/* 🔹 Search & Column Filter */}
            <div className="flex flex-wrap items-center gap-4 mb-4">
                <div className="relative flex items-center">
                    <FaSearch className="absolute left-3 text-gray-400" />
                    <input
                        type="text"
                        placeholder="Search..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        className="input input-bordered pl-10"
                    />
                </div>
                <select
                    className="select select-bordered"
                    value={searchColumn}
                    onChange={(e) => setSearchColumn(e.target.value)}
                >
                    <option value="all">All Columns</option>
                    {columns.map(col => (
                        <option key={col.key} value={col.key}>{col.label}</option>
                    ))}
                </select>
            </div>

            {/* 🔹 Classification Table */}
            <div className="overflow-x-auto">
                <table className="table table-fixed w-full border border-base-300 bg-base-100 rounded-lg shadow-md">
                    <thead>
                    <tr className="bg-base-300 text-base-content">
                        {columns.map(col => (
                            <th
                                key={col.key}
                                onClick={() => handleSort(col.key)}
                                className="cursor-pointer px-3 py-2"
                            >
                                <div className="flex items-center gap-2">
                                    {col.label}
                                    {sortColumn === col.key ? (
                                        sortOrder === "asc" ? <FaSortUp className="text-primary" /> : <FaSortDown className="text-primary" />
                                    ) : (
                                        <FaSort className="text-gray-400" />
                                    )}
                                </div>
                            </th>
                        ))}
                    </tr>
                    </thead>
                    <tbody>
                    {paginatedData.length === 0 ? (
                        <tr>
                            <td colSpan="6" className="text-center p-4 text-gray-500">
                                No results found
                            </td>
                        </tr>
                    ) : (
                        paginatedData.map((row, index) => (
                            <tr key={index} className="hover:bg-base-200 transition">
                                {columns.map(col => (
                                    <td
                                        key={col.key}
                                        className={`p-3 ${col.key === "sample_id" ? "max-w-[200px] truncate" : ""}`}
                                        title={row[col.key]}
                                    >
                                        {row[col.key]}
                                    </td>
                                ))}
                            </tr>
                        ))
                    )}
                    </tbody>
                </table>
            </div>

            {/* 🔹 Pagination Controls */}
            {sortedData.length > 0 && (
                <div className="flex justify-between items-center mt-4">
                    <button
                        className="btn btn-outline btn-sm"
                        disabled={currentPage === 1}
                        onClick={() => setCurrentPage(currentPage - 1)}
                    >
                        « Prev
                    </button>
                    <span className="text-sm font-medium">
                        Page {currentPage} of {totalPages}
                    </span>
                    <button
                        className="btn btn-outline btn-sm"
                        disabled={currentPage >= totalPages}
                        onClick={() => setCurrentPage(currentPage + 1)}
                    >
                        Next »
                    </button>
                </div>
            )}
        </div>
    );
};

export default Plot4Table;
