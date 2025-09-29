import React, { useEffect, useRef } from "react";
import * as d3 from "d3";
import { useSelector } from "react-redux";

const ExamplePlot1 = () => {
    const plots = useSelector((s) => s.visualizationExample.plots);
    const svgRef = useRef();
    const tooltipRef = useRef(null);

    useEffect(() => {
        const points = plots["cs_plot1"];
        if (!points || points.length === 0) return;

        const data = points;

        // Extract row and column labels
        const rowLabels = Array.from(new Set(data.map((d) => d.rowLabel)));
        const colLabels = Array.from(new Set(data.map((d) => d.colLabel)));

        // Fixed container size (adjust as needed or make responsive)
        const containerWidth = 800;
        const containerHeight = 650;
        const margin = { top: 20, right: 160, bottom: 100, left: 120 };
        const width = containerWidth - margin.left - margin.right;
        const height = containerHeight - margin.top - margin.bottom;

        // Clear previous render
        d3.select(svgRef.current).selectAll("*").remove();

        // Root svg + group
        const svg = d3
            .select(svgRef.current)
            .attr("viewBox", `0 0 ${containerWidth} ${containerHeight}`)
            .attr("preserveAspectRatio", "xMidYMid meet")
            .append("g")
            .attr("transform", `translate(${margin.left},${margin.top})`);

        const colorScale = d3.scaleSequential(d3.interpolateViridis).domain([-1, 1]);

        // Scales
        const xScale = d3.scaleBand().domain(colLabels).range([0, width]).padding(0.05);
        const yScale = d3.scaleBand().domain(rowLabels).range([0, height]).padding(0.05);

        // Tooltip (singleton)
        if (!tooltipRef.current) {
            tooltipRef.current = d3
                .select("body")
                .append("div")
                .attr("class", "tooltip bg-gray-900 text-white p-3 rounded-lg shadow-lg text-xs")
                .style("position", "absolute")
                .style("pointer-events", "none")
                .style("display", "none")
                .style("z-index", "1000")
                .style("border", "1px solid #ccc")
                .style("background", "rgba(0, 0, 0, 0.85)");
        }
        const tooltip = tooltipRef.current;

        // Cells
        svg
            .selectAll("rect")
            .data(data)
            .enter()
            .append("rect")
            .attr("x", (d) => xScale(d.colLabel))
            .attr("y", (d) => yScale(d.rowLabel))
            .attr("width", xScale.bandwidth())
            .attr("height", yScale.bandwidth())
            .attr("fill", (d) => colorScale(d.valueLabel))
            .attr("stroke", "#fff")
            .on("mouseover", function (event, d) {
                tooltip
                    .style("display", "block")
                    .style("left", event.pageX + 10 + "px")
                    .style("top", event.pageY - 10 + "px")
                    .html(
                        `<strong>Row:</strong> ${d.rowLabel} <br/>
             <strong>Col:</strong> ${d.colLabel} <br/>
             <strong>Correlation:</strong> ${Math.round(d.valueLabel * 100000) / 100000}`
                    );
                d3.select(this).attr("stroke", "black").attr("stroke-width", 2);
            })
            .on("mousemove", function (event) {
                tooltip.style("left", event.pageX + 10 + "px").style("top", event.pageY - 10 + "px");
            })
            .on("mouseout", function () {
                tooltip.style("display", "none");
                d3.select(this).attr("stroke", "none");
            });

        // Axes
        svg
            .append("g")
            .attr("transform", `translate(0, ${height})`)
            .call(d3.axisBottom(xScale))
            .selectAll("text")
            .attr("transform", "rotate(-45)")
            .style("text-anchor", "end")
            .style("font-size", "12px");

        svg
            .append("g")
            .call(d3.axisLeft(yScale))
            .selectAll("text")
            .style("font-size", "12px");

        // Legend
        const legendWidth = 20;
        const legendHeight = height * 0.8;
        const legendX = width + 50;
        const legendY = height * 0.1;

        const legendScale = d3.scaleLinear().domain(colorScale.domain()).range([legendHeight, 0]);
        const legendAxis = d3.axisRight(legendScale).ticks(5).tickFormat(d3.format(".2f"));

        const defs = svg.append("defs");
        const linearGradient = defs
            .append("linearGradient")
            .attr("id", "legend-gradient-example")
            .attr("x1", "0%")
            .attr("y1", "100%")
            .attr("x2", "0%")
            .attr("y2", "0%");

        linearGradient
            .selectAll("stop")
            .data([
                { offset: "0%", color: colorScale(-1) },
                { offset: "50%", color: colorScale(0) },
                { offset: "100%", color: colorScale(1) },
            ])
            .enter()
            .append("stop")
            .attr("offset", (d) => d.offset)
            .attr("stop-color", (d) => d.color);

        svg
            .append("rect")
            .attr("x", legendX)
            .attr("y", legendY)
            .attr("width", legendWidth)
            .attr("height", legendHeight)
            .style("fill", "url(#legend-gradient-example)");

        svg.append("g").attr("transform", `translate(${legendX + legendWidth}, ${legendY})`).call(legendAxis);

        svg
            .append("text")
            .attr("x", legendX - 10)
            .attr("y", legendY - 10)
            .attr("font-size", "14px")
            .attr("fill", "black")
            .text("Correlation");

        return () => {
            // clean up SVG contents
            d3.select(svgRef.current).selectAll("*").remove();
        };
    }, [plots]);

    return <svg ref={svgRef} id="bc-example-plot1"></svg>;
};

export default ExamplePlot1;
