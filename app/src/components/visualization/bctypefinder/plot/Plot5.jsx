import React, { useEffect, useRef, useMemo } from "react";
import { useSelector } from "react-redux";
import * as d3 from "d3";

const Plot5 = () => {
    const plots = useSelector(state => state.visualization.plots);
    const svgRef = useRef();

    const { data, p_value } = useMemo(() => plots["bc_plot5"] || { data: [], p_value: null }, [plots]);

    const computeKM = (data, maxTime) => {
        const groupedData = d3.group(data, d => d.subtype);
        let survivalData = [];

        for (const [subtype, records] of groupedData) {
            records.sort((a, b) => a.os_time - b.os_time);
            let total = records.length;
            let survival = 1;
            let kmPoints = [];

            // Add start point manually
            kmPoints.push({ time: 0, survival: 1, subtype, censored: false });

            records.forEach((d) => {
                if (d.os_event === 1) {
                    survival *= (total - 1) / total;
                }

                let time = d.os_time;
                if (time === maxTime) time -= 1e-3;

                kmPoints.push({ time, survival, subtype, censored: d.os_event === 0 });
                total--;
            });

            survivalData.push({ subtype, kmPoints });
        }

        return survivalData;
    };

    useEffect(() => {
        if (!data || data.length === 0) return;

        const margin = { top: 20, right: 200, bottom: 80, left: 100 };
        const legendWidth = 100;
        const width = 750 - margin.left - margin.right;
        const height = 450 - margin.top - margin.bottom;

        const maxTime = d3.max(data, d => d.os_time);

        const svg = d3.select(svgRef.current);
        svg.select("g").remove();
        d3.select(".km-tooltip").remove();

        const container = svg
            .attr("viewBox", `0 0 ${width + margin.left + margin.right + legendWidth} ${height + margin.top + margin.bottom}`)
            .attr("preserveAspectRatio", "xMidYMid meet")
            .append("g")
            .attr("transform", `translate(${margin.left},${margin.top})`);

        const tooltip = d3.select("body")
            .append("div")
            .attr("class", "km-tooltip")
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

        const survivalData = computeKM(data, maxTime);
        const sampleCounts = Object.fromEntries(d3.group(data, d => d.subtype).entries().map(([k, v]) => [k, v.length]));

        const xScale = d3.scaleLinear()
            .domain([0, maxTime * 1.02])
            .range([0, width]);

        const yScale = d3.scaleLinear()
            .domain([0, 1])
            .range([height, 0]);

        const colorScale = d3.scaleOrdinal(d3.schemeCategory10);

        container.append("g")
            .attr("transform", `translate(0,${height})`)
            .call(d3.axisBottom(xScale).ticks(6).tickFormat(d3.format(",")))
            .selectAll("text")
            .style("font-size", "12px");

        container.append("g").call(d3.axisLeft(yScale));

        container.append("text")
            .attr("x", width / 2)
            .attr("y", height + 50)
            .attr("text-anchor", "middle")
            .attr("font-size", "14px")
            .text("Time");

        container.append("text")
            .attr("transform", "rotate(-90)")
            .attr("x", -height / 2)
            .attr("y", -60)
            .attr("text-anchor", "middle")
            .attr("font-size", "14px")
            .text("Survival Probability");

        const line = d3.line()
            .x(d => xScale(d.time))
            .y(d => yScale(d.survival))
            .curve(d3.curveStepAfter);

        container.selectAll(".km-line")
            .data(survivalData)
            .enter()
            .append("path")
            .attr("class", d => `km-line line-${d.subtype.replace(/\s/g, "")}`)
            .attr("fill", "none")
            .attr("stroke", d => colorScale(d.subtype))
            .attr("stroke-width", 2.5)
            .attr("d", d => line(d.kmPoints))
            .style("transition", "opacity 0.3s");

        // Draw censor marks and hitboxes
        const censoredPoints = survivalData.flatMap(d => d.kmPoints.filter(p => p.censored));

        // Transparent hover hitboxes (invisible but bigger)
        container.selectAll(".censor-hitbox")
            .data(censoredPoints)
            .enter()
            .append("circle")
            .attr("cx", d => xScale(d.time))
            .attr("cy", d => yScale(d.survival))
            .attr("r", 8)
            .attr("fill", "transparent")
            .on("mouseover", (event, d) => {
                tooltip
                    .style("display", "block")
                    .style("left", `${event.pageX + 10}px`)
                    .style("top", `${event.pageY - 10}px`)
                    .html(
                        `<strong>Subtype:</strong> ${d.subtype}<br>` +
                        `<strong>Time:</strong> ${d.time}<br>` +
                        `<strong>Survival:</strong> ${d.survival.toFixed(5)}`
                    );
            })
            .on("mousemove", (event) => {
                tooltip
                    .style("left", `${event.pageX + 10}px`)
                    .style("top", `${event.pageY - 10}px`);
            })
            .on("mouseout", () => tooltip.style("display", "none"));

        // P-value
        if (p_value !== null) {
            const formatted = p_value.toExponential(1);
            const match = formatted.match(/^([\d.]+)e([-+]?)(\d+)$/);
            if (match) {
                const [_, base, sign, expDigits] = match;
                const exponent = `${sign === "-" ? "−" : ""}${expDigits}`;
                const pText = container.append("text")
                    .attr("x", width * 0.05)
                    .attr("y", height * 0.85)
                    .attr("font-size", "16px")
                    .attr("font-weight", "bold")
                    .attr("fill", "black");

                pText.append("tspan").text(`p = ${base} × 10`);
                pText.append("tspan")
                    .text(exponent)
                    .attr("baseline-shift", "super")
                    .attr("font-size", "10px");
            }
        }

        // Legend with sample count and hover highlight
        const legend = container.append("g")
            .attr("transform", `translate(${width + 20}, 0)`);

        survivalData.forEach((d, i) => {
            const subtype = d.subtype;
            const legendItem = legend.append("g")
                .attr("transform", `translate(0, ${i * 25})`)
                .style("cursor", "pointer")
                .on("mouseover", () => {
                    svg.selectAll(".km-line").style("opacity", l => l.subtype === subtype ? 1 : 0.15);
                    svg.selectAll(".km-tooltip").style("opacity", l => l.subtype === subtype ? 1 : 0.15);
                    svg.selectAll("circle").style("opacity", c => c.subtype === subtype ? 1 : 0.15);
                    svg.selectAll(".censor-hitbox").style("opacity", c => c.subtype === subtype ? 1 : 0.15);
                })
                .on("mouseout", () => {
                    svg.selectAll(".km-line").style("opacity", 1);
                    svg.selectAll(".km-tooltip").style("opacity", 1);
                    svg.selectAll("circle").style("opacity", 1);
                    svg.selectAll(".censor-hitbox").style("opacity", 1);
                });

            legendItem.append("circle")
                .attr("cx", 0)
                .attr("cy", 6)
                .attr("r", 6)
                .attr("fill", colorScale(subtype));

            legendItem.append("text")
                .attr("x", 15)
                .attr("y", 10)
                .attr("font-size", "12px")
                .attr("fill", "#000")
                .text(`${subtype} (n = ${sampleCounts[subtype]})`);
        });

        return () => {
            svg.select("g").remove();
            d3.select(".km-tooltip").remove();
        };
    }, [data, p_value]);

    if (!data || data.length === 0) {
        return (
            <div className="text-center text-gray-500 text-lg py-4">
                No data available for this plot.
            </div>
        );
    }
    return <svg ref={svgRef} id="bc-plot5"></svg>;
};

export default Plot5;
