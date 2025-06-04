import { configureStore } from "@reduxjs/toolkit";
import toastReducer from "./redux/toastSlice.jsx";
import authReducer from "./redux/authSlice.jsx";
import projectReducer from "./redux/projectSlice.jsx";
import jobReducer from "./redux/jobSlice.jsx";
import visualizationReducer from "./redux/visualizationSlice.jsx";


export const store = configureStore({
    reducer: {
        auth: authReducer,
        toast: toastReducer,
        project: projectReducer,
        job: jobReducer,
        visualization: visualizationReducer,
    },
});

export default store;
