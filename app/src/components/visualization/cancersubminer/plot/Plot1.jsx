import React, { useEffect, useRef } from "react";
import * as d3 from "d3";
import { useSelector } from "react-redux";

const Plot1 = () => {
    const plots = useSelector(state => state.visualization.plots);
    const svgRef = useRef();
    const tooltipRef = useRef(null);

    useEffect(() => {
        if (!plots["cs_plot1"] || plots["cs_plot1"].length === 0) return;

        const data = plots["cs_plot1"];

        // Extract row and column labels
        const rowLabels = Array.from(new Set(data.map(d => d.rowLabel)));
        const colLabels = Array.from(new Set(data.map(d => d.colLabel)));

        // Set container dimensions based on parent
        const containerWidth = 800; // Adjust based on the parent div
        const containerHeight = 650; // Adjust based on the parent div
        const margin = { top: 20, right: 160, bottom: 100, left: 120 };
        const width = containerWidth - margin.left - margin.right;
        const height = containerHeight - margin.top - margin.bottom;

        // Clear previous SVG before re-rendering
        d3.select(svgRef.current).selectAll("*").remove();

        // Create SVG container
        const svg = d3.select(svgRef.current)
            .attr("viewBox", `0 0 ${containerWidth} ${containerHeight}`)
            .attr("preserveAspectRatio", "xMidYMid meet")
            .append("g")
            .attr("transform", `translate(${margin.left},${margin.top})`);

        const colorScale = d3.scaleSequential(d3.interpolateViridis)
            .domain([-1, 1]);

        // X and Y scales
        const xScale = d3.scaleBand().domain(colLabels).range([0, width]).padding(0.05);
        const yScale = d3.scaleBand().domain(rowLabels).range([0, height]).padding(0.05);

        // ** Tooltip Setup **
        if (!tooltipRef.current) {
            tooltipRef.current = d3.select("body").append("div")
                .attr("class", "tooltip bg-gray-900 text-white p-3 rounded-lg shadow-lg text-xs")
                .style("position", "absolute")
                .style("pointer-events", "none")
                .style("display", "none")
                .style("z-index", "1000")
                .style("border", "1px solid #ccc")
                .style("background", "rgba(0, 0, 0, 0.85)");
        }

        const tooltip = tooltipRef.current;

        // ** Draw Heatmap Cells **
        svg.selectAll("rect")
            .data(data)
            .enter()
            .append("rect")
            .attr("x", d => xScale(d.colLabel))
            .attr("y", d => yScale(d.rowLabel))
            .attr("width", xScale.bandwidth())
            .attr("height", yScale.bandwidth())
            .attr("fill", d => colorScale(d.valueLabel))
            .attr("stroke", "#fff")
            .on("mouseover", function (event, d) {
                tooltip.style("display", "block")
                    .style("left", (event.pageX + 10) + "px")
                    .style("top", (event.pageY - 10) + "px")
                    .html(`<strong>Row:</strong> ${d.rowLabel} <br>
                           <strong>Col:</strong> ${d.colLabel} <br>
                           <strong>Correlation:</strong> ${Math.round(d.valueLabel * 100000) / 100000}`);

                d3.select(this)
                    .attr("stroke", "black")
                    .attr("stroke-width", 2);
            })
            .on("mousemove", function (event) {
                tooltip.style("left", (event.pageX + 10) + "px")
                    .style("top", (event.pageY - 10) + "px");
            })
            .on("mouseout", function () {
                tooltip.style("display", "none");
                d3.select(this)
                    .attr("stroke", "none");
            });

        // ** X-axis (Column Labels) **
        svg.append("g")
            .attr("transform", `translate(0, ${height})`)
            .call(d3.axisBottom(xScale))
            .selectAll("text")
            .attr("transform", "rotate(-45)")
            .style("text-anchor", "end")
            .style("font-size", "12px");

        // ** Y-axis (Row Labels) **
        svg.append("g")
            .call(d3.axisLeft(yScale))
            .selectAll("text")
            .style("font-size", "12px");

        // ** Axis Titles **
        svg.append("text")
            .attr("x", width / 2)
            .attr("y", height + 90)
            .attr("text-anchor", "middle")
            .attr("font-size", "14px")
            .attr("fill", "black");

        svg.append("text")
            .attr("x", -height / 2)
            .attr("y", -60)
            .attr("text-anchor", "middle")
            .attr("font-size", "14px")
            .attr("fill", "black")
            .attr("transform", "rotate(-90)");

        // ** Improved Legend **
        const legendWidth = 20;
        const legendHeight = height * 0.8;
        const legendX = width + 50;
        const legendY = height * 0.1;

        const legendScale = d3.scaleLinear()
            .domain(colorScale.domain())
            .range([legendHeight, 0]);

        const legendAxis = d3.axisRight(legendScale)
            .ticks(5)
            .tickFormat(d3.format(".2f"));

        // Create legend gradient
        const defs = svg.append("defs");
        const linearGradient = defs.append("linearGradient")
            .attr("id", "legend-gradient")
            .attr("x1", "0%")
            .attr("y1", "100%")
            .attr("x2", "0%")
            .attr("y2", "0%");

        linearGradient.selectAll("stop")
            .data([
                { offset: "0%", color: colorScale(-1) },
                { offset: "50%", color: colorScale(0) },
                { offset: "100%", color: colorScale(1) }
            ])
            .enter()
            .append("stop")
            .attr("offset", d => d.offset)
            .attr("stop-color", d => d.color);

        svg.append("rect")
            .attr("x", legendX)
            .attr("y", legendY)
            .attr("width", legendWidth)
            .attr("height", legendHeight)
            .style("fill", "url(#legend-gradient)");

        svg.append("g")
            .attr("transform", `translate(${legendX + legendWidth}, ${legendY})`)
            .call(legendAxis);

        // ** Legend Title **
        svg.append("text")
            .attr("x", legendX - 10)
            .attr("y", legendY - 10)
            .attr("font-size", "14px")
            .attr("fill", "black")
            .text("Correlation");

        return () => {
            svg.select("g").remove();
        };
    }, [plots]);

    return <svg ref={svgRef} id={"cs-plot1"}></svg>
};

export default Plot1;
