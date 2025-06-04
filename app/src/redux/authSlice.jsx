import {createAsyncThunk, createSlice, isAnyOf} from "@reduxjs/toolkit";
import axios from "axios";
import api from "../shared/utils/axiosInstance.jsx";

const api_url = import.meta.env.VITE_API_BASE_URL;

export const loginUser = createAsyncThunk("auth/login", async (userData, { rejectWithValue }) => {
    return axios.post(
        `${api_url}/auth/login`,
        userData,
        )
        .then((res) => {
            return res.data;
        })
        .catch((err) => {
            return rejectWithValue(err.response?.data?.detail)
        });
});

export const signupUser = createAsyncThunk("auth/signup", async (userData, { rejectWithValue }) => {
    return axios.post(
        `${api_url}/auth/signup`,
        userData
        )
        .then((res) => {
            return res.data;
        })
        .catch((err) => {
            return rejectWithValue(err.response?.data?.detail)
        })
});

export const fetchCurrentUser = createAsyncThunk("auth/me", async (_, { rejectWithValue }) => {
    return api.get(`${api_url}/auth/me`)
        .then((res) => {
            return res.data;
        })
        .catch((err) => {
            return rejectWithValue(err.response?.data?.detail)
        })
})

const authSlice = createSlice({
    name: "auth",
    initialState: {
        user: null,
        token: localStorage.getItem("access_token") || null,
        status: "idle",
        error: null,
    },
    reducers: {
        logout: (state) => {
            state.user = null;
            state.token = null;
            localStorage.removeItem("access_token");
            localStorage.removeItem("refresh_token");
        },
    },
    extraReducers: (builder) => {
        builder
            .addCase(fetchCurrentUser.fulfilled, (state, action) => {
                state.user = action.payload;
                state.status = 'succeeded';
                state.token = localStorage.getItem("access_token");
            })
            .addMatcher(
                isAnyOf(
                    loginUser.pending,
                    signupUser.pending,
                    fetchCurrentUser.pending
                ), (state) =>{
                    state.status = "loading";
                    state.error = null;
                }
            )
            .addMatcher(
                isAnyOf(
                    loginUser.fulfilled,
                    signupUser.fulfilled
                ), (state, action) => {
                    state.status = 'succeeded';
                    state.token = action?.payload.access_token;
                    state.user = action?.payload.user;
                    localStorage.setItem('access_token', action?.payload.access_token);
                    localStorage.setItem('refresh_token', action?.payload.refresh_token);
                })
            .addMatcher(
                isAnyOf(
                    loginUser.rejected,
                    signupUser.rejected,
                    fetchCurrentUser.rejected
                ), (state, action) =>{
                    state.status = "failed";
                    state.error = action?.payload;
                    state.user = null;
                }
            )
    }
});

export const { logout } = authSlice.actions;
export default authSlice.reducer;
