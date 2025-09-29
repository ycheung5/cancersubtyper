import React, { useEffect, useRef, useMemo } from "react";
import { useSelector } from "react-redux";
import * as d3 from "d3";

const ExamplePlot5 = () => {
    const plots = useSelector((s) => s.visualizationExample.plots);
    const { data, p_value } = useMemo(
        () => plots["bc_plot5"] || { data: [], p_value: null },
        [plots]
    );

    const svgRef = useRef();

    // KM computation identical to real plot
    const computeKM = (rows, maxTime) => {
        const grouped = d3.group(rows, (d) => d.subtype);
        const survivalData = [];

        for (const [subtype, recs] of grouped) {
            recs.sort((a, b) => a.os_time - b.os_time);
            let atRisk = recs.length;
            let S = 1;
            const kmPoints = [{ time: 0, survival: 1, subtype, censored: false }];

            recs.forEach((r) => {
                if (r.os_event === 1) S *= (atRisk - 1) / atRisk;
                let t = r.os_time;
                if (t === maxTime) t -= 1e-3; // avoid step drawing at boundary
                kmPoints.push({ time: t, survival: S, subtype, censored: r.os_event === 0 });
                atRisk--;
            });

            survivalData.push({ subtype, kmPoints });
        }
        return survivalData;
    };

    useEffect(() => {
        if (!data?.length) return;

        const margin = { top: 20, right: 200, bottom: 80, left: 100 };
        const legendWidth = 100;
        const width = 750 - margin.left - margin.right;
        const height = 450 - margin.top - margin.bottom;

        const maxTime = d3.max(data, (d) => d.os_time) || 0;

        const svg = d3.select(svgRef.current);
        svg.select("g").remove();
        d3.select(".km-tooltip-example").remove();

        const g = svg
            .attr(
                "viewBox",
                `0 0 ${width + margin.left + margin.right + legendWidth} ${height + margin.top + margin.bottom}`
            )
            .attr("preserveAspectRatio", "xMidYMid meet")
            .append("g")
            .attr("transform", `translate(${margin.left},${margin.top})`);

        const tooltip = d3
            .select("body")
            .append("div")
            .attr("class", "km-tooltip-example")
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

        const kmData = computeKM(data, maxTime);
        const counts = Object.fromEntries(
            d3.group(data, (d) => d.subtype).entries().map(([k, v]) => [k, v.length])
        );

        const x = d3.scaleLinear().domain([0, maxTime * 1.02]).range([0, width]);
        const y = d3.scaleLinear().domain([0, 1]).range([height, 0]);
        const color = d3.scaleOrdinal(d3.schemeCategory10);

        g.append("g").attr("transform", `translate(0,${height})`).call(d3.axisBottom(x).ticks(6));
        g.append("g").call(d3.axisLeft(y));

        g.append("text")
            .attr("x", width / 2)
            .attr("y", height + 50)
            .attr("text-anchor", "middle")
            .attr("font-size", "14px")
            .text("Time");

        g.append("text")
            .attr("transform", "rotate(-90)")
            .attr("x", -height / 2)
            .attr("y", -60)
            .attr("text-anchor", "middle")
            .attr("font-size", "14px")
            .text("Survival Probability");

        const line = d3
            .line()
            .x((d) => x(d.time))
            .y((d) => y(d.survival))
            .curve(d3.curveStepAfter);

        g.selectAll(".km-line")
            .data(kmData)
            .enter()
            .append("path")
            .attr("class", (d) => `km-line line-${d.subtype.replace(/\s/g, "")}`)
            .attr("fill", "none")
            .attr("stroke", (d) => color(d.subtype))
            .attr("stroke-width", 2.5)
            .attr("d", (d) => line(d.kmPoints));

        // Censor hover hitboxes
        const censored = kmData.flatMap((d) => d.kmPoints.filter((p) => p.censored));
        g.selectAll(".censor-hitbox")
            .data(censored)
            .enter()
            .append("circle")
            .attr("cx", (d) => x(d.time))
            .attr("cy", (d) => y(d.survival))
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
                tooltip.style("left", `${event.pageX + 10}px`).style("top", `${event.pageY - 10}px`);
            })
            .on("mouseout", () => tooltip.style("display", "none"));

        // p-value
        if (p_value !== null) {
            const formatted = p_value.toExponential(1);
            const m = formatted.match(/^([\d.]+)e([-+]?)(\d+)$/);
            if (m) {
                const [, base, sign, expDigits] = m;
                const exponent = `${sign === "-" ? "−" : ""}${expDigits}`;
                const pText = g
                    .append("text")
                    .attr("x", width * 0.05)
                    .attr("y", height * 0.85)
                    .attr("font-size", "16px")
                    .attr("font-weight", "bold")
                    .attr("fill", "black");

                pText.append("tspan").text(`p = ${base} × 10`);
                pText
                    .append("tspan")
                    .text(exponent)
                    .attr("baseline-shift", "super")
                    .attr("font-size", "10px");
            }
        }

        // legend
        const legend = g.append("g").attr("transform", `translate(${width + 20}, 0)`);
        kmData.forEach((d, i) => {
            const s = d.subtype;
            const item = legend
                .append("g")
                .attr("transform", `translate(0, ${i * 25})`)
                .style("cursor", "pointer")
                .on("mouseover", () => {
                    svg.selectAll(".km-line").style("opacity", (L) => (L.subtype === s ? 1 : 0.15));
                    svg.selectAll(".censor-hitbox").style("opacity", (C) => (C.subtype === s ? 1 : 0.15));
                })
                .on("mouseout", () => {
                    svg.selectAll(".km-line").style("opacity", 1);
                    svg.selectAll(".censor-hitbox").style("opacity", 1);
                });

            item.append("circle").attr("cx", 0).attr("cy", 6).attr("r", 6).attr("fill", color(s));
            item
                .append("text")
                .attr("x", 15)
                .attr("y", 10)
                .attr("font-size", "12px")
                .attr("fill", "#000")
                .text(`${s} (n = ${counts[s]})`);
        });

        return () => {
            svg.select("g").remove();
            d3.select(".km-tooltip-example").remove();
        };
    }, [data, p_value]);

    if (!data?.length) {
        return (
            <div className="text-center text-gray-500 text-lg py-4">
                No data available for this plot.
            </div>
        );
    }

    return <svg ref={svgRef} id="bc-example-plot5"></svg>;
};

export default ExamplePlot5;
