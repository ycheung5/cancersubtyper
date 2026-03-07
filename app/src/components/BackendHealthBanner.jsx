import React, { useEffect, useState } from "react";
import { FaArrowRotateRight, FaCircleExclamation, FaPlugCircleXmark } from "react-icons/fa6";
import { useLocation, useNavigate } from "react-router-dom";
import { RouteConstants } from "../shared/constants/RouteConstants";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
const HEALTHCHECK_INTERVAL_MS = 30000;

const BackendHealthBanner = () => {
    const [status, setStatus] = useState("checking");
    const [checkedAt, setCheckedAt] = useState(null);
    const navigate = useNavigate();
    const location = useLocation();

    const checkBackendHealth = async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/`, {
                method: "GET",
                headers: { Accept: "application/json" },
            });

            if (!response.ok) {
                throw new Error(`Health check failed with status ${response.status}`);
            }

            setStatus("healthy");
            setCheckedAt(new Date());
        } catch (error) {
            setStatus("unhealthy");
            setCheckedAt(new Date());
        }
    };

    useEffect(() => {
        checkBackendHealth();

        const intervalId = window.setInterval(() => {
            checkBackendHealth();
        }, HEALTHCHECK_INTERVAL_MS);

        return () => window.clearInterval(intervalId);
    }, []);

    useEffect(() => {
        if (status === "unhealthy" && location.pathname !== RouteConstants.home) {
            navigate(RouteConstants.home, { replace: true });
        }
    }, [location.pathname, navigate, status]);

    if (status === "healthy") {
        return null;
    }

    const statusLabel = status === "checking" ? "Checking backend status" : "Backend unavailable";
    const message =
        status === "checking"
            ? "The app is waiting for the API to respond. On the first setup, this can take a while because the backend may still be installing dependencies."
            : "The frontend cannot reach the API right now. On the first setup, the backend may still be installing dependencies. Login, project loading, and job submission will fail until the backend is up.";

    return (
        <div className="fixed inset-x-0 top-0 z-[100]">
            <div className="border-b border-warning/30 bg-base-100/95 shadow-2xl backdrop-blur">
                <div className="mx-auto flex min-h-24 max-w-7xl flex-col gap-4 px-4 py-4 sm:px-6 lg:flex-row lg:items-center lg:justify-between">
                    <div className="flex items-start gap-4">
                        <div className="mt-1 flex h-11 w-11 shrink-0 items-center justify-center rounded-2xl bg-warning/20 text-warning">
                            {status === "checking" ? <FaCircleExclamation className="h-5 w-5" /> : <FaPlugCircleXmark className="h-5 w-5" />}
                        </div>
                        <div>
                            <div className="text-base font-semibold text-base-content">{statusLabel}</div>
                            <div className="mt-1 text-sm text-base-content/75">{message}</div>
                            <div className="mt-2 text-xs text-base-content/60">
                                API: <span className="font-mono">{API_BASE_URL}</span>
                                {checkedAt ? ` · Last checked ${checkedAt.toLocaleTimeString()}` : ""}
                            </div>
                        </div>
                    </div>
                    <button
                        type="button"
                        className="btn btn-warning btn-sm sm:btn-md"
                        onClick={checkBackendHealth}
                    >
                        <FaArrowRotateRight />
                        Retry now
                    </button>
                </div>
            </div>
        </div>
    );
};

export default BackendHealthBanner;
