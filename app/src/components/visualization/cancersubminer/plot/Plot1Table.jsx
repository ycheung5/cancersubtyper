import React, { useEffect, useState, useMemo } from "react";
import { useSelector } from "react-redux";
import { FaSearch, FaSort, FaSortUp, FaSortDown, FaChevronLeft, FaChevronRight } from "react-icons/fa";

const Plot1Table = () => {
    const { plots } = useSelector(state => state.visualization);
    const data = useMemo(() => plots["cs_plot1_table"] || [], [plots]);

    const [loadingState, setLoadingState] = useState("loading");
    const [searchQuery, setSearchQuery] = useState("");
    const [searchColumn, setSearchColumn] = useState("all");
    const [sortColumn, setSortColumn] = useState(null);
    const [sortOrder, setSortOrder] = useState("asc");
    const [currentPage, setCurrentPage] = useState(1);
    const rowsPerPage = 10;

    const columns = [
        { key: "cluster", label: "Cluster" },
        { key: "cpg", label: "CpG" },
        { key: "position", label: "Position" },
        { key: "chr", label: "Chromosome" },
        { key: "strand", label: "Strand" },
        { key: "ucsc", label: "UCSC Gene" },
        { key: "genome", label: "Genome Build" },
    ];

    useEffect(() => {
        if (data.length === 0) return setLoadingState("loading");
        setLoadingState("idle");
    }, [data]);

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
    const paginatedData = useMemo(() => sortedData.slice((currentPage - 1) * rowsPerPage, currentPage * rowsPerPage), [sortedData, currentPage]);

    useEffect(() => {
        if (currentPage > totalPages) setCurrentPage(1);
    }, [totalPages, currentPage]);

    if (loadingState === "loading") return <p className="text-yellow-500">Loading CpG table data...</p>;

    return (
        <div className="p-5">
            {/* 🔹 Search & Filter Section */}
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
                    value={searchColumn}
                    onChange={(e) => setSearchColumn(e.target.value)}
                    className="select select-bordered"
                >
                    <option value="all">All Columns</option>
                    {columns.map(col => <option key={col.key} value={col.key}>{col.label}</option>)}
                </select>
            </div>

            {/* 🔹 Table */}
            <div className="overflow-x-auto">
                <table className="table table-fixed w-full border border-base-300 bg-base-100 rounded-lg shadow-md">
                    <thead>
                    <tr className="bg-base-300 text-base-content">
                        {columns.map(col => (
                            <th
                                key={col.key}
                                className="cursor-pointer"
                                onClick={() => {
                                    setSortColumn(col.key);
                                    setSortOrder(prev => (sortColumn === col.key && prev === "asc") ? "desc" : "asc");
                                }}
                            >
                                <div className="flex items-center gap-2">
                                    {col.label}
                                    {sortColumn === col.key ? (
                                        sortOrder === "asc" ? <FaSortUp /> : <FaSortDown />
                                    ) : (
                                        <FaSort />
                                    )}
                                </div>
                            </th>
                        ))}
                    </tr>
                    </thead>
                    <tbody>
                    {paginatedData.map((row, index) => (
                        <tr key={index} className="hover:bg-base-200">
                            {columns.map(col => (
                                <td
                                    key={col.key}
                                    className={`p-3 ${col.key === "ucsc" ? "max-w-[200px] truncate" : ""}`}
                                    title={row[col.key]}
                                >
                                    {row[col.key]}
                                </td>
                            ))}
                        </tr>
                    ))}
                    </tbody>
                </table>
            </div>

            {/* 🔹 Pagination Controls */}
            <div className="flex justify-between items-center mt-4">
                <button
                    className="btn btn-sm btn-outline"
                    onClick={() => setCurrentPage(prev => Math.max(prev - 1, 1))}
                    disabled={currentPage === 1}
                >
                    <FaChevronLeft /> Prev
                </button>

                <span className="text-sm font-medium">
                    Page {currentPage} of {totalPages}
                </span>

                <button
                    className="btn btn-sm btn-outline"
                    onClick={() => setCurrentPage(prev => Math.min(prev + 1, totalPages))}
                    disabled={currentPage === totalPages}
                >
                    Next <FaChevronRight />
                </button>
            </div>
        </div>
    );
};

export default Plot1Table;
