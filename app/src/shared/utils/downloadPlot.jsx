export const downloadSVG = (svgId, filename = "plot.svg") => {
    const svg = document.getElementById(svgId);
    if (!svg) return;

    const serializer = new XMLSerializer();
    const source = serializer.serializeToString(svg);
    const blob = new Blob([source], { type: "image/svg+xml;charset=utf-8" });

    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = filename;
    link.click();
}

export const downloadPNG = (svgId, filename = "plot.png", scale = 2) => {
    const svg = document.getElementById(svgId);
    if (!svg) return;

    // Clone SVG so we don’t affect the original
    const clone = svg.cloneNode(true);

    // Convert foreignObject to plain SVG <g> fallback for PNG rendering
    const foreignObjects = clone.querySelectorAll("foreignObject");
    foreignObjects.forEach(foreign => {
        const x = +foreign.getAttribute("x") || 0;
        const y = +foreign.getAttribute("y") || 0;

        const div = foreign.querySelector("div");
        if (!div) return;

        const lines = Array.from(div.querySelectorAll("div")).map(el => el.innerText);
        const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
        group.setAttribute("transform", `translate(${x},${y})`);

        lines.forEach((text, i) => {
            const t = document.createElementNS("http://www.w3.org/2000/svg", "text");
            t.setAttribute("x", 0);
            t.setAttribute("y", i * 14);
            t.setAttribute("font-size", "10");
            t.textContent = text;
            group.appendChild(t);
        });

        foreign.replaceWith(group);
    });

    const svgData = new XMLSerializer().serializeToString(clone);
    const canvas = document.createElement("canvas");
    const rect = svg.getBoundingClientRect();
    canvas.width = rect.width * scale;
    canvas.height = rect.height * scale;

    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    const img = new Image();
    const svgBlob = new Blob([svgData], { type: "image/svg+xml;charset=utf-8" });
    const url = URL.createObjectURL(svgBlob);

    img.onload = () => {
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        URL.revokeObjectURL(url);
        const pngUrl = canvas.toDataURL("image/png");

        const link = document.createElement("a");
        link.href = pngUrl;
        link.download = filename;
        link.click();
    };

    img.src = url;
};
