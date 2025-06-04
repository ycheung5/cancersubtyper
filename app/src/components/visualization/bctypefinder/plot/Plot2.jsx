import React, { useEffect, useMemo, useRef } from "react";
import { useSelector } from "react-redux";
import * as d3 from "d3";

const Plot2 = () => {
    const plots = useSelector(state => state.visualization.plots);
    const svgRef = useRef();
    const tooltipRef = useRef(null);
    const data = useMemo(() => plots["bc_plot2"] || [], [plots]);

    useEffect(() => {
        if (data.length === 0) return;

        const margin = { top: 50, right: 200, bottom: 100, left: 90 };
        const width = 800 - margin.left - margin.right;
        const height = 500 - margin.top - margin.bottom;

        d3.select(svgRef.current).selectAll("*").remove();

        const svg = d3.select(svgRef.current)
            .attr("width", width + margin.left + margin.right)
            .attr("height", height + margin.top + margin.bottom)
            .append("g")
            .attr("transform", `translate(${margin.left},${margin.top})`);

        const groupedData = d3.group(data, d => d.subtype);

        const boxData = Array.from(groupedData, ([subtype, values]) => {
            const sorted = values.map(d => d.value).sort(d3.ascending);
            const q1 = d3.quantile(sorted, 0.25);
            const median = d3.quantile(sorted, 0.5);
            const q3 = d3.quantile(sorted, 0.75);
            return {
                subtype,
                q1,
                median,
                q3,
                min: d3.min(sorted),
                max: d3.max(sorted),
                values,
                count: values.length
            };
        });

        const xScale = d3.scaleBand()
            .domain(boxData.map(d => d.subtype))
            .range([0, width])
            .padding(0.2);

        const yScale = d3.scaleLinear()
            .domain([0, d3.max(data, d => d.value)])
            .nice()
            .range([height, 0]);

        const colorScale = d3.scaleOrdinal(d3.schemeCategory10);

        svg.append("g")
            .attr("transform", `translate(0,${height})`)
            .call(d3.axisBottom(xScale))
            .selectAll("text")
            .attr("transform", "rotate(-25)")
            .style("text-anchor", "end");

        svg.append("g")
            .call(d3.axisLeft(yScale).ticks(6));

        // ✅ Floating tooltip
        if (!tooltipRef.current) {
            tooltipRef.current = d3.select("body").append("div")
                .attr("class", "tooltip")
                .style("position", "absolute")
                .style("pointer-events", "none")
                .style("background", "#111")
                .style("color", "#fff")
                .style("padding", "8px 12px")
                .style("border-radius", "6px")
                .style("font-size", "12px")
                .style("box-shadow", "0 2px 6px rgba(0,0,0,0.3)")
                .style("display", "none")
                .style("z-index", "1000");
        }

        const tooltip = tooltipRef.current;

        // ✅ Draw boxes
        svg.selectAll(".box")
            .data(boxData)
            .enter()
            .append("rect")
            .attr("x", d => xScale(d.subtype))
            .attr("y", d => yScale(d.q3))
            .attr("width", xScale.bandwidth())
            .attr("height", d => yScale(d.q1) - yScale(d.q3))
            .attr("fill", d => colorScale(d.subtype))
            .attr("stroke", "black")
            .on("mouseover", function (event, d) {
                tooltip.style("display", "block")
                    .style("left", `${event.pageX + 10}px`)
                    .style("top", `${event.pageY - 10}px`)
                    .html(`<strong>Subtype:</strong> ${d.subtype}<br/>
                           <strong>n:</strong> ${d.count}<br/>
                           <strong>Q1:</strong> ${d.q1.toFixed(2)}<br/>
                           <strong>Median:</strong> ${d.median.toFixed(2)}<br/>
                           <strong>Q3:</strong> ${d.q3.toFixed(2)}`);
                d3.select(this).attr("stroke-width", 2);
            })
            .on("mousemove", event => {
                tooltip.style("left", `${event.pageX + 10}px`)
                    .style("top", `${event.pageY - 10}px`);
            })
            .on("mouseout", function () {
                tooltip.style("display", "none");
                d3.select(this).attr("stroke-width", 1);
            });

        // ✅ Draw additional lines (whiskers + median)
        svg.selectAll(".median-line")
            .data(boxData)
            .enter()
            .append("line")
            .attr("x1", d => xScale(d.subtype))
            .attr("x2", d => xScale(d.subtype) + xScale.bandwidth())
            .attr("y1", d => yScale(d.median))
            .attr("y2", d => yScale(d.median))
            .attr("stroke", "black")
            .attr("stroke-width", 2);

        svg.selectAll(".whiskers")
            .data(boxData)
            .enter()
            .append("g")
            .each(function (d) {
                const xMid = xScale(d.subtype) + xScale.bandwidth() / 2;
                const capW = xScale.bandwidth() / 2.5;

                // vertical lines
                d3.select(this).append("line")
                    .attr("x1", xMid).attr("x2", xMid)
                    .attr("y1", yScale(d.min)).attr("y2", yScale(d.q1))
                    .attr("stroke", "black");

                d3.select(this).append("line")
                    .attr("x1", xMid).attr("x2", xMid)
                    .attr("y1", yScale(d.q3)).attr("y2", yScale(d.max))
                    .attr("stroke", "black");

                // caps
                d3.select(this).append("line")
                    .attr("x1", xMid - capW / 2).attr("x2", xMid + capW / 2)
                    .attr("y1", yScale(d.min)).attr("y2", yScale(d.min))
                    .attr("stroke", "black");

                d3.select(this).append("line")
                    .attr("x1", xMid - capW / 2).attr("x2", xMid + capW / 2)
                    .attr("y1", yScale(d.max)).attr("y2", yScale(d.max))
                    .attr("stroke", "black");
            });

        // ✅ Legend
        const legend = svg.append("g").attr("transform", `translate(${width + 30}, 10)`);

        boxData.forEach((d, i) => {
            const yOffset = i * 25;

            const legendGroup = legend.append("g")
                .attr("transform", `translate(0, ${yOffset})`)
                .style("cursor", "pointer")
                .on("mouseover", () => {
                    svg.selectAll(".box").style("opacity", b => b.subtype === d.subtype ? 1 : 0.2);
                })
                .on("mouseout", () => {
                    svg.selectAll(".box").style("opacity", 1);
                });

            legendGroup.append("circle")
                .attr("cx", 0)
                .attr("cy", 6)
                .attr("r", 6)
                .attr("fill", colorScale(d.subtype));

            legendGroup.append("text")
                .attr("x", 15)
                .attr("y", 10)
                .text(`${d.subtype} (n = ${d.count})`)
                .attr("font-size", "12px")
                .attr("fill", "#000");
        });

        return () => {
            d3.select(svgRef.current).selectAll("*").remove();
            tooltip.remove();
        };
    }, [data]);

    return <svg ref={svgRef} id={"bc-plot2"}></svg>;
};

export default Plot2;
