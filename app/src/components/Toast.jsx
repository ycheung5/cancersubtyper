import React, { useEffect, useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import { hideToast } from "../redux/toastSlice";
import { XCircleIcon, CheckCircleIcon } from "@heroicons/react/24/solid";

const Toast = () => {
    const dispatch = useDispatch();
    const { message, type, visible } = useSelector((state) => state.toast);
    const [timeoutId, setTimeoutId] = useState(null);

    useEffect(() => {
        if (visible) {
            const id = setTimeout(() => {
                dispatch(hideToast());
            }, 3000);
            setTimeoutId(id);
        }

        return () => {
            if (timeoutId) clearTimeout(timeoutId);
        };
    }, [visible, dispatch]);

    const handleMouseEnter = () => {
        if (timeoutId) clearTimeout(timeoutId);
    };

    const handleMouseLeave = () => {
        const id = setTimeout(() => {
            dispatch(hideToast());
        }, 2000);
        setTimeoutId(id);
    };

    if (!visible) return null;

    return (
        <div
            className={`fixed bottom-6 right-6 z-[9999] transition-all duration-300 transform ${
                visible ? "opacity-100 scale-100" : "opacity-0 scale-95"
            }`}
            onMouseEnter={handleMouseEnter}
            onMouseLeave={handleMouseLeave}
        >
            <div
                className={`alert shadow-lg ${
                    type === "success" ? "alert-success" : "alert-error"
                } w-[calc(100vw-2rem)] max-w-xl items-start px-4 py-3 rounded-lg`}
            >
                <div className="flex items-center gap-3">
                    {type === "success" ? (
                        <CheckCircleIcon className="mt-0.5 h-6 w-6 shrink-0 text-white" />
                    ) : (
                        <XCircleIcon className="mt-0.5 h-6 w-6 shrink-0 text-white" />
                    )}
                    <span className="break-words pr-2 text-white">
                        {message}
                    </span>
                </div>
            </div>
        </div>
    );
};

export default Toast;
