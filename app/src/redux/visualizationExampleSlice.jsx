import { createAsyncThunk, createSlice } from "@reduxjs/toolkit";
import api from "../shared/utils/axiosInstance.jsx";

/**
 * Models: "bctypefinder" | "cancersubminer"
 * Example routes do NOT require job_id.
 *
 * Endpoints used:
 *  - /visualization/examples/{model}/plot1-options
 *  - /visualization/examples/{model}/plot1/{batch}/{subtype}
 *  - /visualization/examples/{model}/plot1-table/{clusters}
 *  - /visualization/examples/{model}/plot2-options
 *  - /visualization/examples/{model}/plot2/{option}/{batch}
 *  - /visualization/examples/{model}/plot3/{option}
 *  - /visualization/examples/cancersubminer/plot3-kmean
 *  - /visualization/examples/cancersubminer/plot3-nemo
 *  - /visualization/examples/{model}/plot4-table
 *  - /visualization/examples/{model}/plot5-options
 *  - /visualization/examples/{model}/plot5/{batch}
 */

// ------------------------ BCtypeFinder (examples) ----------------------------

// Plot 1
export const getBCExamplePlot1Option = createAsyncThunk(
    "vizExample/getBCPlot1Option",
    async (_, { rejectWithValue }) => {
        return api
            .get(`/visualization/examples/bctypefinder/plot1-options`)
            .then((res) => res.data)
            .catch((err) => rejectWithValue(err.response?.data?.detail));
    }
);

export const getBCExamplePlot1 = createAsyncThunk(
    "vizExample/getBCPlot1",
    async ({ batch, subtype }, { rejectWithValue }) => {
        return api
            .get(
                `/visualization/examples/bctypefinder/plot1/${encodeURIComponent(
                    batch
                )}/${encodeURIComponent(subtype)}`
            )
            .then((res) => res.data)
            .catch((err) => rejectWithValue(err.response?.data?.detail));
    }
);

export const getBCExamplePlot1Table = createAsyncThunk(
    "vizExample/getBCPlot1Table",
    async ({ clusters }, { rejectWithValue }) => {
        return api
            .get(
                `/visualization/examples/bctypefinder/plot1-table/${encodeURIComponent(
                    clusters
                )}`
            )
            .then((res) => res.data)
            .catch((err) => rejectWithValue(err.response?.data?.detail));
    }
);

// Plot 2
export const getBCExamplePlot2Option = createAsyncThunk(
    "vizExample/getBCPlot2Option",
    async (_, { rejectWithValue }) => {
        return api
            .get(`/visualization/examples/bctypefinder/plot2-options`)
            .then((res) => res.data)
            .catch((err) => rejectWithValue(err.response?.data?.detail));
    }
);

export const getBCExamplePlot2 = createAsyncThunk(
    "vizExample/getBCPlot2",
    async ({ option, batch }, { rejectWithValue }) => {
        return api
            .get(
                `/visualization/examples/bctypefinder/plot2/${encodeURIComponent(
                    option
                )}/${encodeURIComponent(batch)}`
            )
            .then((res) => res.data)
            .catch((err) => rejectWithValue(err.response?.data?.detail));
    }
);

// Plot 3
export const getBCExamplePlot3 = createAsyncThunk(
    "vizExample/getBCPlot3",
    async ({ option }, { rejectWithValue }) => {
        return api
            .get(`/visualization/examples/bctypefinder/plot3/${option}`)
            .then((res) => res.data)
            .catch((err) => rejectWithValue(err.response?.data?.detail));
    }
);

// Plot 4 (table)
export const getBCExamplePlot4Table = createAsyncThunk(
    "vizExample/getBCPlot4Table",
    async (_, { rejectWithValue }) => {
        return api
            .get(`/visualization/examples/bctypefinder/plot4-table`)
            .then((res) => res.data)
            .catch((err) => rejectWithValue(err.response?.data?.detail));
    }
);

// Plot 5 (KM)
export const getBCExamplePlot5Option = createAsyncThunk(
    "vizExample/getBCPlot5Option",
    async (_, { rejectWithValue }) => {
        return api
            .get(`/visualization/examples/bctypefinder/plot5-options`)
            .then((res) => res.data)
            .catch((err) => rejectWithValue(err.response?.data?.detail));
    }
);

export const getBCExamplePlot5 = createAsyncThunk(
    "vizExample/getBCPlot5",
    async ({ batch }, { rejectWithValue }) => {
        return api
            .get(
                `/visualization/examples/bctypefinder/plot5/${encodeURIComponent(
                    batch
                )}`
            )
            .then((res) => res.data)
            .catch((err) => rejectWithValue(err.response?.data?.detail));
    }
);

// ------------------------ CancerSubminer (examples) --------------------------

// Plot 1
export const getCSExamplePlot1Option = createAsyncThunk(
    "vizExample/getCSPlot1Option",
    async (_, { rejectWithValue }) => {
        return api
            .get(`/visualization/examples/cancersubminer/plot1-options`)
            .then((res) => res.data)
            .catch((err) => rejectWithValue(err.response?.data?.detail));
    }
);

export const getCSExamplePlot1 = createAsyncThunk(
    "vizExample/getCSPlot1",
    async ({ batch, subtype }, { rejectWithValue }) => {
        return api
            .get(
                `/visualization/examples/cancersubminer/plot1/${encodeURIComponent(
                    batch
                )}/${encodeURIComponent(subtype)}`
            )
            .then((res) => res.data)
            .catch((err) => rejectWithValue(err.response?.data?.detail));
    }
);

export const getCSExamplePlot1Table = createAsyncThunk(
    "vizExample/getCSPlot1Table",
    async ({ clusters }, { rejectWithValue }) => {
        return api
            .get(
                `/visualization/examples/cancersubminer/plot1-table/${encodeURIComponent(
                    clusters
                )}`
            )
            .then((res) => res.data)
            .catch((err) => rejectWithValue(err.response?.data?.detail));
    }
);

// Plot 2
export const getCSExamplePlot2Option = createAsyncThunk(
    "vizExample/getCSPlot2Option",
    async (_, { rejectWithValue }) => {
        return api
            .get(`/visualization/examples/cancersubminer/plot2-options`)
            .then((res) => res.data)
            .catch((err) => rejectWithValue(err.response?.data?.detail));
    }
);

export const getCSExamplePlot2 = createAsyncThunk(
    "vizExample/getCSPlot2",
    async ({ option, batch }, { rejectWithValue }) => {
        return api
            .get(
                `/visualization/examples/cancersubminer/plot2/${encodeURIComponent(
                    option
                )}/${encodeURIComponent(batch)}`
            )
            .then((res) => res.data)
            .catch((err) => rejectWithValue(err.response?.data?.detail));
    }
);

// Plot 3
export const getCSExamplePlot3 = createAsyncThunk(
    "vizExample/getCSPlot3",
    async ({ option }, { rejectWithValue }) => {
        return api
            .get(`/visualization/examples/cancersubminer/plot3/${option}`)
            .then((res) => res.data)
            .catch((err) => rejectWithValue(err.response?.data?.detail));
    }
);
export const getCSExamplePlot3KMean = createAsyncThunk(
    "vizExample/getCSPlot3KMean",
    async (_, { rejectWithValue }) => {
        return api
            .get(`/visualization/examples/cancersubminer/plot3-kmean`)
            .then((res) => res.data)
            .catch((err) => rejectWithValue(err.response?.data?.detail));
    }
);
export const getCSExamplePlot3Nemo = createAsyncThunk(
    "vizExample/getCSPlot3Nemo",
    async (_, { rejectWithValue }) => {
        return api
            .get(`/visualization/examples/cancersubminer/plot3-nemo`)
            .then((res) => res.data)
            .catch((err) => rejectWithValue(err.response?.data?.detail));
    }
);

// Plot 4 (table)
export const getCSExamplePlot4Table = createAsyncThunk(
    "vizExample/getCSPlot4Table",
    async (_, { rejectWithValue }) => {
        return api
            .get(`/visualization/examples/cancersubminer/plot4-table`)
            .then((res) => res.data)
            .catch((err) => rejectWithValue(err.response?.data?.detail));
    }
);

// Plot 5 (KM)
export const getCSExamplePlot5Option = createAsyncThunk(
    "vizExample/getCSPlot5Option",
    async (_, { rejectWithValue }) => {
        return api
            .get(`/visualization/examples/cancersubminer/plot5-options`)
            .then((res) => res.data)
            .catch((err) => rejectWithValue(err.response?.data?.detail));
    }
);

export const getCSExamplePlot5 = createAsyncThunk(
    "vizExample/getCSPlot5",
    async ({ batch }, { rejectWithValue }) => {
        return api
            .get(
                `/visualization/examples/cancersubminer/plot5/${encodeURIComponent(
                    batch
                )}`
            )
            .then((res) => res.data)
            .catch((err) => rejectWithValue(err.response?.data?.detail));
    }
);

// ----------------------------- Slice ----------------------------------------
const visualizationExampleSlice = createSlice({
    name: "visualizationExample",
    initialState: {
        plots: {
            // BC examples
            bc_plot1_option: {},
            bc_plot1: [],
            bc_plot1_table: [],
            bc_plot2_option: { cpg_groups: [], batches: [] },
            bc_plot2: [],
            bc_plot3_corrected: [],
            bc_plot3_uncorrected: [],
            bc_plot4_table: [],
            bc_plot5_option: [],
            bc_plot5: { data: [], p_value: NaN },

            // CS examples
            cs_plot1_option: {},
            cs_plot1: [],
            cs_plot1_table: [],
            cs_plot2_option: { cpg_groups: [], batches: [] },
            cs_plot2: [],
            cs_plot3_corrected: [],
            cs_plot3_uncorrected: [],
            cs_plot3_kmean: [],
            cs_plot3_nemo: [],
            cs_plot4_table: [],
            cs_plot5_option: [],
            cs_plot5: { data: [], p_value: NaN },
        },
        status: "idle",
        error: null,
    },
    reducers: {},
    extraReducers: (builder) => {
        const getPlotKey = (plotKey, action) => {
            // for plot3 option-specific storage
            if (plotKey === "bc_plot3" || plotKey === "cs_plot3") {
                return `${plotKey}_${action.meta.arg.option}`;
            }
            if (plotKey === "cs_plot3_kmean" || plotKey === "cs_plot3_nemo") {
                return `cs_plot3_${plotKey.split("_")[2]}`;
            }
            return plotKey;
        };

        const handleFulfilled = (plotKey) => (state, action) => {
            state.plots[getPlotKey(plotKey, action)] = action.payload;
            state.status = "succeeded";
            state.error = null;
        };

        const handleRejected = (plotKey) => (state, action) => {
            state.plots[getPlotKey(plotKey, action)] = undefined;
            if (plotKey === "bc_plot1") state.plots["bc_plot1_table"] = [];
            if (plotKey === "cs_plot1") state.plots["cs_plot1_table"] = [];
            state.status = "failed";
            state.error = action.payload ?? "Request failed";
        };

        const handlePending = (plotKey) => (state) => {
            state.plots[plotKey] = undefined;
            state.status = "loading";
            state.error = null;
        };

        // --- BC example reducers
        builder
            .addCase(getBCExamplePlot1Option.pending, handlePending("bc_plot1_option"))
            .addCase(getBCExamplePlot1.pending, handlePending("bc_plot1"))
            .addCase(getBCExamplePlot1Table.pending, handlePending("bc_plot1_table"))

            .addCase(getBCExamplePlot1Option.fulfilled, handleFulfilled("bc_plot1_option"))
            .addCase(getBCExamplePlot1.fulfilled, handleFulfilled("bc_plot1"))
            .addCase(getBCExamplePlot1Table.fulfilled, handleFulfilled("bc_plot1_table"))

            .addCase(getBCExamplePlot1Option.rejected, handleRejected("bc_plot1_option"))
            .addCase(getBCExamplePlot1.rejected, handleRejected("bc_plot1"))
            .addCase(getBCExamplePlot1Table.rejected, handleRejected("bc_plot1_table"))

            .addCase(getBCExamplePlot2Option.pending, handlePending("bc_plot2_option"))
            .addCase(getBCExamplePlot2.pending, handlePending("bc_plot2"))
            .addCase(getBCExamplePlot2Option.fulfilled, handleFulfilled("bc_plot2_option"))
            .addCase(getBCExamplePlot2.fulfilled, handleFulfilled("bc_plot2"))
            .addCase(getBCExamplePlot2Option.rejected, handleRejected("bc_plot2_option"))
            .addCase(getBCExamplePlot2.rejected, handleRejected("bc_plot2"))

            .addCase(getBCExamplePlot3.pending, handlePending("bc_plot3"))
            .addCase(getBCExamplePlot3.fulfilled, handleFulfilled("bc_plot3"))
            .addCase(getBCExamplePlot3.rejected, handleRejected("bc_plot3"))

            .addCase(getBCExamplePlot4Table.pending, handlePending("bc_plot4_table"))
            .addCase(getBCExamplePlot4Table.fulfilled, handleFulfilled("bc_plot4_table"))
            .addCase(getBCExamplePlot4Table.rejected, handleRejected("bc_plot4_table"))

            .addCase(getBCExamplePlot5Option.pending, handlePending("bc_plot5_option"))
            .addCase(getBCExamplePlot5.pending, handlePending("bc_plot5"))
            .addCase(getBCExamplePlot5Option.fulfilled, handleFulfilled("bc_plot5_option"))
            .addCase(getBCExamplePlot5.fulfilled, handleFulfilled("bc_plot5"))
            .addCase(getBCExamplePlot5Option.rejected, handleRejected("bc_plot5_option"))
            .addCase(getBCExamplePlot5.rejected, handleRejected("bc_plot5"));

        // --- CS example reducers
        builder
            .addCase(getCSExamplePlot1Option.pending, handlePending("cs_plot1_option"))
            .addCase(getCSExamplePlot1.pending, handlePending("cs_plot1"))
            .addCase(getCSExamplePlot1Table.pending, handlePending("cs_plot1_table"))

            .addCase(getCSExamplePlot1Option.fulfilled, handleFulfilled("cs_plot1_option"))
            .addCase(getCSExamplePlot1.fulfilled, handleFulfilled("cs_plot1"))
            .addCase(getCSExamplePlot1Table.fulfilled, handleFulfilled("cs_plot1_table"))

            .addCase(getCSExamplePlot1Option.rejected, handleRejected("cs_plot1_option"))
            .addCase(getCSExamplePlot1.rejected, handleRejected("cs_plot1"))
            .addCase(getCSExamplePlot1Table.rejected, handleRejected("cs_plot1_table"))

            .addCase(getCSExamplePlot2Option.pending, handlePending("cs_plot2_option"))
            .addCase(getCSExamplePlot2.pending, handlePending("cs_plot2"))
            .addCase(getCSExamplePlot2Option.fulfilled, handleFulfilled("cs_plot2_option"))
            .addCase(getCSExamplePlot2.fulfilled, handleFulfilled("cs_plot2"))
            .addCase(getCSExamplePlot2Option.rejected, handleRejected("cs_plot2_option"))
            .addCase(getCSExamplePlot2.rejected, handleRejected("cs_plot2"))

            .addCase(getCSExamplePlot3.pending, handlePending("cs_plot3"))
            .addCase(getCSExamplePlot3.fulfilled, handleFulfilled("cs_plot3"))
            .addCase(getCSExamplePlot3.rejected, handleRejected("cs_plot3"))

            .addCase(getCSExamplePlot3KMean.pending, handlePending("cs_plot3_kmean"))
            .addCase(getCSExamplePlot3KMean.fulfilled, handleFulfilled("cs_plot3_kmean"))
            .addCase(getCSExamplePlot3KMean.rejected, handleRejected("cs_plot3_kmean"))

            .addCase(getCSExamplePlot3Nemo.pending, handlePending("cs_plot3_nemo"))
            .addCase(getCSExamplePlot3Nemo.fulfilled, handleFulfilled("cs_plot3_nemo"))
            .addCase(getCSExamplePlot3Nemo.rejected, handleRejected("cs_plot3_nemo"))

            .addCase(getCSExamplePlot4Table.pending, handlePending("cs_plot4_table"))
            .addCase(getCSExamplePlot4Table.fulfilled, handleFulfilled("cs_plot4_table"))
            .addCase(getCSExamplePlot4Table.rejected, handleRejected("cs_plot4_table"))

            .addCase(getCSExamplePlot5Option.pending, handlePending("cs_plot5_option"))
            .addCase(getCSExamplePlot5.pending, handlePending("cs_plot5"))
            .addCase(getCSExamplePlot5Option.fulfilled, handleFulfilled("cs_plot5_option"))
            .addCase(getCSExamplePlot5.fulfilled, handleFulfilled("cs_plot5"))
            .addCase(getCSExamplePlot5Option.rejected, handleRejected("cs_plot5_option"))
            .addCase(getCSExamplePlot5.rejected, handleRejected("cs_plot5"));
    },
});

export default visualizationExampleSlice.reducer;
