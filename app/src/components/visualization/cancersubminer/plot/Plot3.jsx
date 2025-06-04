import React, { useEffect, useRef, useMemo } from "react";
import * as d3 from "d3";
import { useSelector } from "react-redux";

const Plot3 = ({ dataset, svgId }) => {
    const plots = useSelector(state => state.visualization.plots);
    const svgRef = useRef();
    const tooltipRef = useRef(null);

    const data = useMemo(() => plots[`cs_plot3_${dataset}`] || [], [plots, dataset]);

    useEffect(() => {
        if (data.length === 0) return;

        const margin = { top: 30, right: 160, bottom: 60, left: 40 };
        const width = 450 - margin.left - margin.right;
        const height = 350 - margin.top - margin.bottom;

        const svg = d3.select(svgRef.current);
        svg.select("g").remove();

        const container = svg
            .attr("viewBox", `0 0 ${width + margin.left + margin.right} ${height + margin.top + margin.bottom}`)
            .attr("preserveAspectRatio", "xMidYMid meet")
            .append("g")
            .attr("transform", `translate(${margin.left},${margin.top})`);

        const xScale = d3.scaleLinear()
            .domain(d3.extent(data, d => d.x)).nice()
            .range([0, width]);

        const yScale = d3.scaleLinear()
            .domain(d3.extent(data, d => d.y)).nice()
            .range([height, 0]);

        const subtypeSet = Array.from(new Set(data.map(d => d.subtype)))
            .sort((a, b) => {
                const numA = parseInt(a.match(/\d+/)?.[0]);
                const numB = parseInt(b.match(/\d+/)?.[0]);
                return numA - numB;
            });

        const subtypeColor = d3.scaleOrdinal(d3.schemeCategory10).domain(subtypeSet);

        const batchSet = Array.from(new Set(data.map(d => d.batch)));
        const symbols = [d3.symbolCircle, d3.symbolTriangle, d3.symbolSquare, d3.symbolCross, d3.symbolDiamond, d3.symbolStar];
        const batchSymbol = d3.scaleOrdinal().domain(batchSet).range(symbols);

        if (!tooltipRef.current) {
            tooltipRef.current = d3.select("body").append("div")
                .attr("class", "tooltip bg-gray-900 text-white p-2 rounded-lg shadow-lg text-sm")
                .style("position", "absolute")
                .style("pointer-events", "none")
                .style("display", "none")
                .style("z-index", "1000");
        }

        const tooltip = tooltipRef.current;

        container.append("g")
            .attr("class", "grid")
            .attr("transform", `translate(0,${height})`)
            .call(d3.axisBottom(xScale).tickSize(-height).tickFormat(""))
            .selectAll("line")
            .style("stroke", "#ccc").style("stroke-opacity", 0.7);

        container.append("g")
            .attr("class", "grid")
            .call(d3.axisLeft(yScale).tickSize(-width).tickFormat(""))
            .selectAll("line")
            .style("stroke", "#ccc").style("stroke-opacity", 0.7);

        container.append("g")
            .attr("transform", `translate(0,${height})`)
            .call(d3.axisBottom(xScale).ticks(6))
            .selectAll("text").style("font-size", "12px");

        container.append("g")
            .call(d3.axisLeft(yScale).ticks(6))
            .selectAll("text").style("font-size", "12px");

        const points = container.selectAll("path.sample")
            .data(data)
            .join("path")
            .attr("class", "sample")
            .attr("transform", d => `translate(${xScale(d.x)},${yScale(d.y)})`)
            .attr("d", d3.symbol().size(40).type(d => batchSymbol(d.batch)))
            .attr("fill", d => subtypeColor(d.subtype))
            .attr("opacity", 0.8)
            .on("mouseover", function (event, d) {
                tooltip.style("display", "block")
                    .style("left", (event.pageX + 10) + "px")
                    .style("top", (event.pageY - 10) + "px")
                    .html(`<strong>Sample ID:</strong> ${d.sample_id} <br>
                           <strong>Batch:</strong> ${d.batch} <br>
                           <strong>Subtype:</strong> ${d.subtype} <br>
                           <strong>x:</strong> ${d.x.toFixed(2)} <br>
                           <strong>y:</strong> ${d.y.toFixed(2)}`);

                d3.select(this)
                    .transition().duration(150)
                    .attr("stroke", "black").attr("stroke-width", 2);
            })
            .on("mouseout", function () {
                tooltip.style("display", "none");
                d3.select(this)
                    .transition().duration(150)
                    .attr("stroke", null).attr("stroke-width", null);
            });

        const truncate = (label, maxLength = 12) => label.length > maxLength ? label.slice(0, maxLength) + "…" : label;

        const legendForeign = container.append("foreignObject")
            .attr("x", width + 30)
            .attr("y", 0)
            .attr("width", 120)
            .attr("height", height);

        const legendDiv = legendForeign.append("xhtml:div")
            .style("max-height", `${height}px`)
            .style("overflow-y", "auto")
            .style("font-family", "sans-serif");

        const subtypeLegend = legendDiv.append("div").style("margin-bottom", "1em");

        subtypeLegend.append("div")
            .style("font-weight", "bold")
            .style("font-size", "11px")
            .style("margin-bottom", "4px")
            .text("Subtype (Color)");

        subtypeSet.forEach((subtype, i) => {
            const row = subtypeLegend.append("div")
                .style("display", "flex")
                .style("align-items", "center")
                .style("margin-bottom", "4px")
                .style("cursor", "pointer")
                .on("mouseover", () => points.transition().duration(200).style("opacity", d => d.subtype === subtype ? 1 : 0.01))
                .on("mouseout", () => points.transition().duration(200).style("opacity", 0.8));

            row.append("div")
                .style("width", "10px")
                .style("height", "10px")
                .style("border-radius", "50%")
                .style("background", subtypeColor(subtype))
                .style("margin-right", "6px");

            row.append("div")
                .style("font-size", "10px")
                .text(truncate(subtype));
        });

        const batchLegend = legendDiv.append("div");

        batchLegend.append("div")
            .style("font-weight", "bold")
            .style("font-size", "11px")
            .style("margin", "8px 0 4px 0")
            .text("Batch (Shape)");

        batchSet.forEach((batch, i) => {
            const row = batchLegend.append("div")
                .style("display", "flex")
                .style("align-items", "center")
                .style("margin-bottom", "4px")
                .style("cursor", "pointer")
                .on("mouseover", () => points.transition().duration(200).style("opacity", d => d.batch === batch ? 1 : 0.01))
                .on("mouseout", () => points.transition().duration(200).style("opacity", 0.8));

            const symbolPath = d3.symbol().type(batchSymbol(batch)).size(64)();

            row.append("svg")
                .attr("width", 12)
                .attr("height", 12)
                .append("path")
                .attr("d", symbolPath)
                .attr("fill", "#666")
                .attr("transform", "translate(6,6)");

            row.append("div")
                .style("margin-left", "6px")
                .style("font-size", "10px")
                .text(truncate(batch));
        });

        return () => svg.select("g").remove();
    }, [data]);

    if (data.length === 0) {
        return (
            <div style={{ textAlign: "center", padding: "20px", fontSize: "18px", fontWeight: "bold", color: "gray" }}>
                No data available for this plot.
            </div>
        );
    }

    return <svg ref={svgRef} id={svgId} className="w-full h-auto"></svg>;
};

export default Plot3;