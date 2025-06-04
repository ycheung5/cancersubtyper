import { createAsyncThunk, createSlice } from "@reduxjs/toolkit";
import api from "../shared/utils/axiosInstance.jsx";

export const getBCPlot1Option = createAsyncThunk(
    "visualization/getBCPlot1Option",
    async ({ job_id }, { rejectWithValue }) => {
        return api
            .get(`/visualization/${job_id}/bctypefinder/plot1-options`)
            .then((res) => res.data)
            .catch((err) => rejectWithValue(err.response?.data?.detail));
    }
);

export const getBCPlot1 = createAsyncThunk(
    "visualization/getBCPlot1",
    async ({ job_id, batch, subtype }, { rejectWithValue }) => {
        return api
            .get(`/visualization/${job_id}/bctypefinder/plot1/${batch}/${subtype}`)
            .then((res) => res.data)
            .catch((err) => rejectWithValue(err.response?.data?.detail));
    }
);

export const getBCPlot1Table = createAsyncThunk(
    "visualization/getBCPlot1Table",
    async ({ job_id, clusters }, { rejectWithValue }) => {
        return api
            .get(`/visualization/${job_id}/bctypefinder/plot1-table/${clusters}`)
            .then((res) => res.data)
            .catch((err) => rejectWithValue(err.response?.data?.detail));
    }
);

export const getBCPlot2Option = createAsyncThunk(
    "visualization/getBCPlot2Option",
    async ({ job_id }, { rejectWithValue }) => {
        return api
            .get(`/visualization/${job_id}/bctypefinder/plot2-options`)
            .then((res) => res.data)
            .catch((err) => rejectWithValue(err.response?.data?.detail));
    }
);

export const getBCPlot2 = createAsyncThunk(
    "visualization/getBCPlot2",
    async ({ job_id, cluster, batch }, { rejectWithValue }) => {
        return api
            .get(`/visualization/${job_id}/bctypefinder/plot2/${cluster}/${batch}`)
            .then((res) => res.data)
            .catch((err) => rejectWithValue(err.response?.data?.detail));
    }
);

export const getBCPlot3 = createAsyncThunk(
    "visualization/getBCPlot3",
    async ({ job_id, option }, { rejectWithValue }) => {
        return api
            .get(`/visualization/${job_id}/bctypefinder/plot3/${option}`)
            .then((res) => res.data)
            .catch((err) => rejectWithValue(err.response?.data?.detail));
    }
);

export const getBCPlot4Table = createAsyncThunk(
    "visualization/getBCPlot4Table",
    async ({ job_id }, { rejectWithValue }) => {
        return api
            .get(`/visualization/${job_id}/bctypefinder/plot4-table`)
            .then((res) => res.data)
            .catch((err) => rejectWithValue(err.response?.data?.detail));
    }
);

export const getBCPlot5Option = createAsyncThunk(
    "visualization/getBCPlot5Option",
    async ({ job_id }, { rejectWithValue }) => {
        return api
            .get(`/visualization/${job_id}/bctypefinder/plot5-options`)
            .then((res) => res.data)
            .catch((err) => rejectWithValue(err.response?.data?.detail));
    }
);

export const getBCPlot5 = createAsyncThunk(
    "visualization/getBCPlot5",
    async ({ job_id, batch }, { rejectWithValue }) => {
        return api
            .get(`/visualization/${job_id}/bctypefinder/plot5/${batch}`)
            .then((res) => res.data)
            .catch((err) => rejectWithValue(err.response?.data?.detail));
    }
);

export const getCSPlot1Option = createAsyncThunk(
    "visualization/getCSPlot1Option",
    async ({ job_id }, { rejectWithValue }) => {
        return api
            .get(`/visualization/${job_id}/cancersubminer/plot1-options`)
            .then((res) => res.data)
            .catch((err) => rejectWithValue(err.response?.data?.detail));
    }
);

export const getCSPlot1 = createAsyncThunk(
    "visualization/getCSPlot1",
    async ({ job_id, batch, subtype }, { rejectWithValue }) => {
        return api
            .get(`/visualization/${job_id}/cancersubminer/plot1/${batch}/${subtype}`)
            .then((res) => res.data)
            .catch((err) => rejectWithValue(err.response?.data?.detail));
    }
);

export const getCSPlot1Table = createAsyncThunk(
    "visualization/getCSPlot1Table",
    async ({ job_id, clusters }, { rejectWithValue }) => {
        return api
            .get(`/visualization/${job_id}/cancersubminer/plot1-table/${clusters}`)
            .then((res) => res.data)
            .catch((err) => rejectWithValue(err.response?.data?.detail));
    }
);

export const getCSPlot2Option = createAsyncThunk(
    "visualization/getCSPlot2Option",
    async ({ job_id }, { rejectWithValue }) => {
        return api
            .get(`/visualization/${job_id}/cancersubminer/plot2-options`)
            .then((res) => res.data)
            .catch((err) => rejectWithValue(err.response?.data?.detail));
    }
);

export const getCSPlot2 = createAsyncThunk(
    "visualization/getCSPlot2",
    async ({ job_id, cluster, batch }, { rejectWithValue }) => {
        return api
            .get(`/visualization/${job_id}/cancersubminer/plot2/${cluster}/${batch}`)
            .then((res) => res.data)
            .catch((err) => rejectWithValue(err.response?.data?.detail));
    }
);

export const getCSPlot3 = createAsyncThunk(
    "visualization/getCSPlot3",
    async ({ job_id, option }, { rejectWithValue }) => {
        return api
            .get(`/visualization/${job_id}/cancersubminer/plot3/${option}`)
            .then((res) => res.data)
            .catch((err) => rejectWithValue(err.response?.data?.detail));
    }
);

export const getCSPlot3KMean = createAsyncThunk(
    "visualization/getPlot3KMean",
    async ({ job_id }, { rejectWithValue }) => {
        return api
            .get(`/visualization/${job_id}/cancersubminer/plot3-kmean`)
            .then((res) => res.data)
            .catch((err) => rejectWithValue(err.response?.data?.detail));
    }
);

export const getCSPlot3Nemo = createAsyncThunk(
    "visualization/getCSPlot3Nemo",
    async ({ job_id }, { rejectWithValue }) => {
        return api
            .get(`/visualization/${job_id}/cancersubminer/plot3-nemo`)
            .then((res) => res.data)
            .catch((err) => rejectWithValue(err.response?.data?.detail));
    }
);

export const getCSPlot4Table = createAsyncThunk(
    "visualization/getCSPlot4Table",
    async ({ job_id }, { rejectWithValue }) => {
        return api
            .get(`/visualization/${job_id}/cancersubminer/plot4-table`)
            .then((res) => res.data)
            .catch((err) => rejectWithValue(err.response?.data?.detail));
    }
)

export const getCSPlot5Option = createAsyncThunk(
    "visualization/getCSPlot5Option",
    async ({ job_id }, { rejectWithValue }) => {
        return api
            .get(`/visualization/${job_id}/cancersubminer/plot5-options`)
            .then((res) => res.data)
            .catch((err) => rejectWithValue(err.response?.data?.detail));
    }
);

export const getCSPlot5 = createAsyncThunk(
    "visualization/getCSPlot5",
    async ({ job_id, batch }, { rejectWithValue }) => {
        return api
            .get(`/visualization/${job_id}/cancersubminer/plot5/${batch}`)
            .then((res) => res.data)
            .catch((err) => rejectWithValue(err.response?.data?.detail));
    }
);

const visualizationSlice = createSlice({
    name: "visualization",
    initialState: {
        plots: {
            bc_plot1_option: {},
            bc_plot1: [],
            bc_plot1_table: [],
            bc_plot2_option: { cpg_groups: [], batches: [] },
            bc_plot2: [],
            bc_plot3: {},
            bc_plot4_table: [],
            bc_plot5_option: [],
            bc_plot5: { data: [], p_value: NaN },

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
        };

        const handlePending = (plotKey) => (state, action) => {
            state.plots[getPlotKey(plotKey, action)] = undefined;
            state.status = "loading";
        };

        builder
            .addCase(getBCPlot1Option.pending, handlePending("bc_plot1_option"))
            .addCase(getBCPlot1.pending, handlePending("bc_plot1"))
            .addCase(getBCPlot1Table.pending, handlePending("bc_plot1_table"))

            .addCase(getBCPlot1Option.fulfilled, handleFulfilled("bc_plot1_option"))
            .addCase(getBCPlot1.fulfilled, handleFulfilled("bc_plot1"))
            .addCase(getBCPlot1Table.fulfilled, handleFulfilled("bc_plot1_table"))

            .addCase(getBCPlot1Option.rejected, handleRejected("bc_plot1_option"))
            .addCase(getBCPlot1.rejected, handleRejected("bc_plot1"))
            .addCase(getBCPlot1Table.rejected, handleRejected("bc_plot1_table"))

            .addCase(getBCPlot2Option.pending, handlePending("bc_plot2_option"))
            .addCase(getBCPlot2.pending, handlePending("bc_plot2"))
            .addCase(getBCPlot2Option.fulfilled, handleFulfilled("bc_plot2_option"))
            .addCase(getBCPlot2.fulfilled, handleFulfilled("bc_plot2"))
            .addCase(getBCPlot2Option.rejected, handleRejected("bc_plot2_option"))
            .addCase(getBCPlot2.rejected, handleRejected("bc_plot2"))

            .addCase(getBCPlot3.pending, handlePending("bc_plot3"))
            .addCase(getBCPlot3.fulfilled, handleFulfilled("bc_plot3"))
            .addCase(getBCPlot3.rejected, handleRejected("bc_plot3"))

            .addCase(getBCPlot4Table.pending, handlePending("bc_plot4_table"))
            .addCase(getBCPlot4Table.fulfilled, handleFulfilled("bc_plot4_table"))
            .addCase(getBCPlot4Table.rejected, handleRejected("bc_plot4_table"))

            .addCase(getBCPlot5Option.pending, handlePending("bc_plot5_option"))
            .addCase(getBCPlot5.pending, handlePending("bc_plot5"))
            .addCase(getBCPlot5Option.fulfilled, handleFulfilled("bc_plot5_option"))
            .addCase(getBCPlot5.fulfilled, handleFulfilled("bc_plot5"))
            .addCase(getBCPlot5Option.rejected, handleRejected("bc_plot5_option"))
            .addCase(getBCPlot5.rejected, handleRejected("bc_plot5"))

            .addCase(getCSPlot1Option.pending, handlePending("cs_plot1_option"))
            .addCase(getCSPlot1.pending, handlePending("cs_plot1"))
            .addCase(getCSPlot1Table.pending, handlePending("cs_plot1_table"))
            .addCase(getCSPlot1Option.fulfilled, handleFulfilled("cs_plot1_option"))
            .addCase(getCSPlot1.fulfilled, handleFulfilled("cs_plot1"))
            .addCase(getCSPlot1Table.fulfilled, handleFulfilled("cs_plot1_table"))
            .addCase(getCSPlot1Option.rejected, handleRejected("cs_plot1_option"))
            .addCase(getCSPlot1.rejected, handleRejected("cs_plot1"))
            .addCase(getCSPlot1Table.rejected, handleRejected("cs_plot1_table"))

            .addCase(getCSPlot2Option.pending, handlePending("cs_plot2_option"))
            .addCase(getCSPlot2.pending, handlePending("cs_plot2"))
            .addCase(getCSPlot2Option.fulfilled, handleFulfilled("cs_plot2_option"))
            .addCase(getCSPlot2.fulfilled, handleFulfilled("cs_plot2"))
            .addCase(getCSPlot2Option.rejected, handleRejected("cs_plot2_option"))
            .addCase(getCSPlot2.rejected, handleRejected("cs_plot2"))

            .addCase(getCSPlot3.pending, handlePending("cs_plot3"))
            .addCase(getCSPlot3.fulfilled, handleFulfilled("cs_plot3"))
            .addCase(getCSPlot3.rejected, handleRejected("cs_plot3"))

            .addCase(getCSPlot3KMean.pending, handlePending("cs_plot3_kmean"))
            .addCase(getCSPlot3KMean.fulfilled, handleFulfilled("cs_plot3_kmean"))
            .addCase(getCSPlot3KMean.rejected, handleRejected("cs_plot3_kmean"))

            .addCase(getCSPlot3Nemo.pending, handlePending("cs_plot3_nemo"))
            .addCase(getCSPlot3Nemo.fulfilled, handleFulfilled("cs_plot3_nemo"))
            .addCase(getCSPlot3Nemo.rejected, handleRejected("cs_plot3_nemo"))

            .addCase(getCSPlot4Table.pending, handlePending("cs_plot4_table"))
            .addCase(getCSPlot4Table.fulfilled, handleFulfilled("cs_plot4_table"))
            .addCase(getCSPlot4Table.rejected, handleRejected("cs_plot4_table"))

            .addCase(getCSPlot5Option.pending, handlePending("cs_plot5_option"))
            .addCase(getCSPlot5.pending, handlePending("cs_plot5"))
            .addCase(getCSPlot5Option.fulfilled, handleFulfilled("cs_plot5_option"))
            .addCase(getCSPlot5.fulfilled, handleFulfilled("cs_plot5"))
            .addCase(getCSPlot5Option.rejected, handleRejected("cs_plot5_option"))
            .addCase(getCSPlot5.rejected, handleRejected("cs_plot5"));
    },
});

export default visualizationSlice.reducer;