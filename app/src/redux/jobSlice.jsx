import { createAsyncThunk, createSlice, isAnyOf } from "@reduxjs/toolkit";
import api from "../shared/utils/axiosInstance.jsx";

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
            .addMatcher(
                isAnyOf(getJobList.pending, createJob.pending, downloadResults.pending),
                (state) => {
                    state.status = 'loading';
                }
            )
            .addMatcher(
                isAnyOf(getJobList.rejected, createJob.rejected, downloadResults.rejected),
                (state, action) => {
                    state.status = 'failed';
                    state.error = action?.payload;
                }
            );
    }
});

export default jobSlice.reducer;
