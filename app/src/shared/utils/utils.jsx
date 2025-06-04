import jsSHA from "jssha";

export const truncateString = (length, string) => {
    if (string.length > length) {
        return string.substring(0, length) + "...";
    }
    return string;
};

export const computeChecksum = async (file, cancelSignal) => {
    const chunkSize = 1024 * 1024 * 1; // 1MB
    const shaObj = new jsSHA("SHA-256", "ARRAYBUFFER");
    let offset = 0;
    const reader = new FileReader();

    return new Promise((resolve, reject) => {
        if (cancelSignal?.aborted) return reject(new Error("Checksum cancelled"));

        reader.onerror = () => reject(new Error("Error reading file."));
        reader.onload = (e) => {
            if (cancelSignal?.aborted) {
                reader.abort();
                return reject(new Error("Checksum cancelled"));
            }

            shaObj.update(new Uint8Array(e.target.result));
            offset += chunkSize;

            const percent = ((offset / file.size) * 100).toFixed(2);
            // console.log(`Checksum Progress: ${percent}%`);

            offset < file.size ? readNextChunk() : resolve(shaObj.getHash("HEX"));
        };

        const readNextChunk = () => {
            if (cancelSignal?.aborted) {
                reader.abort();
                return reject(new Error("Checksum cancelled"));
            }
            reader.readAsArrayBuffer(file.slice(offset, offset + chunkSize));
        };

        readNextChunk();
    });
};

export const formatDate = (date) => {
    return new Date(date).toLocaleString(undefined, {
        year: "numeric",
        month: "2-digit",
        day: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
        hour12: true
    })
};

export const capitalizeFirstLetter = (str) => {
    if (!str) return "";
    return str.charAt(0).toUpperCase() + str.slice(1);
};