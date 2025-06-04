import {createAsyncThunk, createSlice, isAnyOf} from "@reduxjs/toolkit";
import api from "../shared/utils/axiosInstance.jsx";
import {computeChecksum} from "../shared/utils/utils.jsx";
import axios from "axios";

export const getProjects = createAsyncThunk("project/list", async (_, { rejectWithValue }) => {
    return api.get("/project")
        .then((res) => {
            return res.data;
        })
        .catch((err) => {
            return rejectWithValue(err.response?.data?.detail);
        })
});

export const getStorageInfo = createAsyncThunk("project/storage", async (_, { rejectWithValue }) => {
    return api.get(`/storage`)
        .then((res) => {
            return res.data;
        })
        .catch((err) => {
            return rejectWithValue(err.response?.data?.detail);
        })
});

export const createProject = createAsyncThunk("project/create", async (projectData, { rejectWithValue }) => {
    return api.post(`/project/create`, projectData)
        .then((res) => {
            return res.data;
        })
        .catch((err) => {
            return rejectWithValue(err.response?.data?.detail);
        })
});

export const uploadMetadataFile = createAsyncThunk("project/uploadMetadata", async ({ project_id, metadataFile }, { rejectWithValue }) => {
    const formData = new FormData();

    return computeChecksum(metadataFile)
        .then((checksum) => {
            formData.append("metadata", metadataFile);
            formData.append("metadata_checksum", checksum);
            return api.put(`/project/${project_id}/upload_metadata`, formData,
                {headers: { "Content-Type": "multipart/form-data" }});
        })
        .then((res) => {
            return res.data;
        })
        .catch((err) => {
            return rejectWithValue(err.response?.data?.detail);
        });
    }
);

export const updateProject = createAsyncThunk(
    "project/edit",
    async ({ projectId, projectData }, { rejectWithValue }) => {
        return api
            .put(`/project/${projectId}`, projectData)
            .then((res) => res.data)
            .catch((err) => rejectWithValue(err.response?.data?.detail));
    }
);


export const uploadSampleFile = createAsyncThunk(
    "project/uploadSample",
    async ({ project_id, sourceFile, targetFile, setProgress, cancelTokenSource, abortSignal }, { rejectWithValue }) => {
        try {
            const sourceChecksum = await computeChecksum(sourceFile, abortSignal);
            const targetChecksum = await computeChecksum(targetFile, abortSignal);

            const formData = new FormData();
            formData.append("source", sourceFile);
            formData.append("target", targetFile);
            formData.append("source_checksum", sourceChecksum);
            formData.append("target_checksum", targetChecksum);

            const res = await api.put(`/project/${project_id}/upload`, formData, {
                headers: { "Content-Type": "multipart/form-data" },
                onUploadProgress: (e) => {
                    const percent = Math.round((e.loaded * 100) / e.total);
                    setProgress(percent);
                },
                cancelToken: cancelTokenSource?.token,
            });

            return res.data;
        } catch (err) {
            if (abortSignal?.aborted || axios.isCancel(err)) {
                return rejectWithValue("Upload cancelled");
            }
            return rejectWithValue(err.response?.data?.detail || "Upload failed");
        }
    }
);

export const deleteProject = createAsyncThunk("project/delete", async (project_id, { rejectWithValue }) => {
    return api.delete(`/project/${project_id}`)
        .then(() => project_id)
        .catch((err) => rejectWithValue(err.response?.data?.detail));
});

const projectSlice = createSlice({
    name: "project",
    initialState: {
        projectList: [],
        storageInfo: null,
        tumorTypes: [],
        status: "idle",
        error: null,
    },
    reducers: {},
    extraReducers: (builder) => {
        builder
            .addCase(getProjects.fulfilled, (state, action) => {
                state.status = "succeeded";
                state.projectList = action.payload
            })
            .addCase(getStorageInfo.fulfilled, (state, action) => {
                state.storageInfo = action.payload;
            })
            .addCase(createProject.fulfilled, (state, action) => {
                state.status = "succeeded";
                state.projectList.push(action.payload);
            })
            .addCase(uploadMetadataFile.fulfilled, (state, action) => {
                state.status = "succeeded";
                const index = state.projectList.findIndex(project => project.id === action?.payload?.id);
                if (index !== -1) {
                    state.projectList[index] = action.payload;
                } else {
                    state.projectList.push(action.payload);
                }
            })
            .addCase(uploadSampleFile.fulfilled, (state, action) => {
                state.status = "succeeded";
                const index = state.projectList.findIndex(project => project.id === action?.payload?.id);
                if (index !== -1) {
                    state.projectList[index] = action.payload;
                } else {
                    state.projectList.push(action.payload);
                }
            })
            .addCase(deleteProject.fulfilled, (state, action) => {
                state.projectList = state.projectList.filter(project => project.id !== action.payload);
            })
            .addCase(updateProject.fulfilled, (state, action) => {
                state.status = "succeeded";
                const index = state.projectList.findIndex(project => project.id === action.payload.id);
                if (index !== -1) {
                    state.projectList[index] = action.payload;
                }
            })
            .addMatcher(
                isAnyOf(
                    getProjects.pending,
                    createProject.pending,
                    uploadMetadataFile.pending,
                    uploadSampleFile.pending,
                    deleteProject.pending,
                    updateProject.pending,
                ), (state) => {
                    state.status = "loading"
                }
            )
            .addMatcher(
                isAnyOf(
                    getProjects.rejected,
                ), (state, action) => {
                    state.status = "failed"
                    state.error = action.payload
                }
            )
    }
})

export default projectSlice.reducer;