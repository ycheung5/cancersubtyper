import React, { useEffect, useRef, useMemo } from "react";
import * as d3 from "d3";
import { useSelector } from "react-redux";

const Plot3 = ({ dataset, group, svgId }) => {
    const plots = useSelector(state => state.visualization.plots);
    const svgRef = useRef();
    const tooltipRef = useRef(null);

    const data = useMemo(() => plots[`bc_plot3_${dataset}`] || [], [plots, dataset]);

    useEffect(() => {
        if (data.length === 0) return;

        const margin = { top: 30, right: 120, bottom: 60, left: 70 };
        const width = 450 - margin.left - margin.right;
        const height = 350 - margin.top - margin.bottom;

        const svg = d3.select(svgRef.current);
        svg.select("g").remove();

        const container = svg
            .attr("viewBox", `0 0 ${width + margin.left + margin.right} ${height + margin.top + margin.bottom}`)
            .attr("preserveAspectRatio", "xMidYMid meet")
            .append("g")
            .attr("transform", `translate(${margin.left},${margin.top})`);

        // Dynamically compute scales
        const xScale = d3.scaleLinear()
            .domain(d3.extent(data, d => d.x))
            .nice()
            .range([0, width]);

        const yScale = d3.scaleLinear()
            .domain(d3.extent(data, d => d.y))
            .nice()
            .range([height, 0]);

        const colorScale = d3.scaleOrdinal(d3.schemeCategory10);

        // ** Tooltip Styling **
        if (!tooltipRef.current) {
            tooltipRef.current = d3.select("body").append("div")
                .attr("class", "tooltip bg-gray-900 text-white p-2 rounded-lg shadow-lg text-sm")
                .style("position", "absolute")
                .style("pointer-events", "none")
                .style("display", "none")
                .style("z-index", "1000");
        }

        const tooltip = tooltipRef.current;

        // ** Gridlines **
        container.append("g")
            .attr("class", "grid")
            .attr("transform", `translate(0,${height})`)
            .call(d3.axisBottom(xScale).tickSize(-height).tickFormat(""))
            .selectAll("line")
            .style("stroke", "#ccc")
            .style("stroke-opacity", 0.7);

        container.append("g")
            .attr("class", "grid")
            .call(d3.axisLeft(yScale).tickSize(-width).tickFormat(""))
            .selectAll("line")
            .style("stroke", "#ccc")
            .style("stroke-opacity", 0.7);

        // ** X and Y Axes **
        container.append("g")
            .attr("transform", `translate(0,${height})`)
            .call(d3.axisBottom(xScale).ticks(6))
            .selectAll("text")
            .style("font-size", "12px");

        container.append("g")
            .call(d3.axisLeft(yScale).ticks(6))
            .selectAll("text")
            .style("font-size", "12px");

        // ** Scatter Points **
        container.selectAll("circle")
            .data(data)
            .join("circle")
            .attr("cx", d => xScale(d.x))
            .attr("cy", d => yScale(d.y))
            .attr("r", 2)
            .attr("fill", d => colorScale(d[group]))
            .attr("opacity", 0.8)
            .attr("stroke", "none")
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
                    .attr("stroke", "black")
                    .attr("stroke-width", 2)
                    .attr("r", 5);
            })
            .on("mouseout", function () {
                tooltip.style("display", "none");

                d3.select(this)
                    .transition().duration(150)
                    .attr("stroke", "none")
                    .attr("r", 2);
            });

        // ** Legend **
        const groups = [...new Set(data.map(d => d[group]))];

        const legend = container.append("g")
            .attr("transform", `translate(${width + 30}, 20)`);

        // ** Legend Dots **
        legend.selectAll(".legend-dot")
            .data(groups)
            .join("circle")
            .attr("cx", 0)
            .attr("cy", (d, i) => i * 22)
            .attr("r", 6)
            .attr("fill", d => colorScale(d))
            .style("cursor", "pointer")
            .on("mouseover", function (event, hoveredGroup) {
                container.selectAll("circle")
                    .transition().duration(200)
                    .style("opacity", d => d[group] === hoveredGroup ? 1 : 0.2);
            })
            .on("mouseout", function () {
                container.selectAll("circle")
                    .transition().duration(200)
                    .style("opacity", 0.8);
            });

        // ** Legend Labels **
        legend.selectAll(".legend-label")
            .data(groups)
            .join("text")
            .attr("x", 15)
            .attr("y", (d, i) => i * 22 + 5)
            .text(d => d)
            .attr("font-size", "12px")
            .attr("fill", "black")
            .style("cursor", "pointer")
            .on("mouseover", function (event, hoveredGroup) {
                container.selectAll("circle")
                    .transition().duration(200)
                    .style("opacity", d => d[group] === hoveredGroup ? 1 : 0.2);
            })
            .on("mouseout", function () {
                container.selectAll("circle")
                    .transition().duration(200)
                    .style("opacity", 0.8);
            });

        return () => {
            svg.select("g").remove();
        };
    }, [data, group]);

    if (data.length === 0) {
        return (
            <div style={{ textAlign: "center", padding: "20px", fontSize: "18px", fontWeight: "bold", color: "gray" }}>
                No data available for this plot.
            </div>
        );
    }

    return <svg ref={svgRef} id={svgId} className="w-full h-auto"></svg>
};

export default Plot3;
