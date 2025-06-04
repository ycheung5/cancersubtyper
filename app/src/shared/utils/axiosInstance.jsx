import axios from "axios";
import { RouteConstants } from "../constants/RouteConstants.js";

const base_url = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

const api = axios.create({
    baseURL: base_url,
    headers: { "Content-Type": "application/json" },
});

// Attach Access Token from localStorage
api.interceptors.request.use(config => {
    const token = localStorage.getItem("access_token");
    if (token) config.headers.Authorization = `Bearer ${token}`;
    return config;
});

// Logout Handler (Prevents Circular Dependency)
const handleLogout = () => {
    localStorage.removeItem("access_token");
    localStorage.removeItem("refresh_token");
    window.location.href = RouteConstants.login;
};

// Handle 401 Unauthorized (Token Expiry)
api.interceptors.response.use(
    response => response,
    async error => {
        const originalRequest = error.config;
        const refresh_token = localStorage.getItem("refresh_token");

        if (error.response?.status === 401 && refresh_token && !originalRequest._retry) {
            originalRequest._retry = true;

            try {
                const res = await axios.post(`${base_url}/auth/refresh?refresh_token=${refresh_token}`);

                const newAccessToken = res.data.access_token;
                localStorage.setItem("access_token", newAccessToken);

                api.defaults.headers.Authorization = `Bearer ${newAccessToken}`;
                originalRequest.headers.Authorization = `Bearer ${newAccessToken}`;
                return api(originalRequest);
            } catch (err) {
                handleLogout();
                return Promise.reject(err);
            }
        }

        return Promise.reject(error);
    }
);

export default api;
