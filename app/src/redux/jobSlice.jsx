import { createAsyncThunk, createSlice, isAnyOf } from "@reduxjs/toolkit";
import api from "../shared/utils/axiosInstance.jsx";
import axios from "axios";

const base_url = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export const getJobList = createAsyncThunk('job/list', async (project_id, { rejectWithValue }) => {
    return api.get(`/job/project/${project_id}`)
        .then(res => res.data)
        .catch(err => rejectWithValue(err.response?.data?.detail));
});

export const getJob = createAsyncThunk('job/job', async (job_id, { rejectWithValue }) => {
    return api.get(`/job/${job_id}`)
        .then(res => res.data)
        .catch(err => rejectWithValue(err.response?.data?.detail));
})

export const createJob = createAsyncThunk('job/create', async (jobData, { rejectWithValue }) => {
    return api.post('/job', jobData)
        .then(res => res.data)
        .catch(err => rejectWithValue(err.response?.data?.detail));
});

export const getModelList = createAsyncThunk('job/model', async (job_id, { rejectWithValue }) => {
    return api.get(`/job/model/all-models`)
        .then(res => res.data)
        .catch(err => rejectWithValue(err.response?.data?.detail));
});

export const downloadResults = createAsyncThunk('job/results', async (job_id, { rejectWithValue }) => {
    return api.get(`/job/${job_id}/download/results`, {
        responseType: 'blob',
    })
        .then(res => res.data)
        .catch(err => rejectWithValue(err.response?.data?.detail));
});

export const downloadExampleResults = createAsyncThunk('job/exampleResults', async (model, { rejectWithValue }) => {
    return axios.get(`${base_url}/job/examples/${model}/results`, {
        responseType: 'blob',
    })
        .then(res => ({ data: res.data, model }))
        .catch(err => rejectWithValue(err.response?.data?.detail || 'Failed to download example results'));
});

export const downloadExampleDataset = createAsyncThunk('job/exampleDataset', async (datasetType, { rejectWithValue }) => {
    return axios.get(`${base_url}/job/examples/datasets/${datasetType}`, {
        responseType: 'blob',
    })
        .then(res => ({ data: res.data, datasetType }))
        .catch(err => rejectWithValue(err.response?.data?.detail || 'Failed to download example dataset'));
});

const jobSlice = createSlice({
    name: 'job',
    initialState: {
        jobList: [],
        modelList:[],
        status: 'idle',
        error: null,
    },
    reducers: {},
    extraReducers: (builder) => {
        builder
            .addCase(getJobList.fulfilled, (state, action) => {
                state.status = 'succeeded';
                state.jobList = action.payload;
            })
            .addCase(createJob.fulfilled, (state, action) => {
                state.status = 'succeeded';
                state.jobList.push(action.payload);
            })
            .addCase(getModelList.fulfilled, (state, action) => {
                state.modelList = action.payload;
            })
            .addCase(downloadResults.fulfilled, (state, action) => {
                state.status = 'succeeded';

                const blob = new Blob([action.payload], { type: 'application/zip' });
                const url = window.URL.createObjectURL(blob);

                const a = document.createElement('a');
                a.href = url;
                a.download = 'results.zip';
                document.body.appendChild(a);
                a.click();

                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);
            })
            .addCase(downloadExampleResults.fulfilled, (state, action) => {
                state.status = 'succeeded';

                const blob = new Blob([action.payload.data], { type: 'application/zip' });
                const url = window.URL.createObjectURL(blob);

                const a = document.createElement('a');
                a.href = url;
                a.download = `${action.payload.model}_example_results.zip`;
                document.body.appendChild(a);
                a.click();

                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);
            })
            .addCase(downloadExampleDataset.fulfilled, (state, action) => {
                state.status = 'succeeded';

                // Determine file type and name based on dataset type
                const datasetType = action.payload.datasetType;
                let fileType, fileName;
                
                if (datasetType === 'metadata') {
                    fileType = 'text/csv';
                    fileName = 'example_metadata.csv';
                } else {
                    fileType = 'text/csv';
                    fileName = `example_${datasetType}.csv`;
                }

                const blob = new Blob([action.payload.data], { type: fileType });
                const url = window.URL.createObjectURL(blob);

                const a = document.createElement('a');
                a.href = url;
                a.download = fileName;
                document.body.appendChild(a);
                a.click();

                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);
            })
            .addMatcher(
                isAnyOf(getJobList.pending, createJob.pending, downloadResults.pending, downloadExampleResults.pending, downloadExampleDataset.pending),
                (state) => {
                    state.status = 'loading';
                }
            )
            .addMatcher(
                isAnyOf(getJobList.rejected, createJob.rejected, downloadResults.rejected, downloadExampleResults.rejected, downloadExampleDataset.rejected),
                (state, action) => {
                    state.status = 'failed';
                    state.error = action?.payload;
                }
            );
    }
});

export default jobSlice.reducer;
