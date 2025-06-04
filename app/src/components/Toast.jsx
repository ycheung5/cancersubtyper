import React, { useEffect, useState, useRef } from "react";
import { useDispatch, useSelector } from "react-redux";
import { hideToast } from "../redux/toastSlice";
import { XCircleIcon, CheckCircleIcon, XMarkIcon } from "@heroicons/react/24/solid";

const Toast = () => {
    const dispatch = useDispatch();
    const { message, type, visible } = useSelector((state) => state.toast);
    const [timeoutId, setTimeoutId] = useState(null);
    const messageRef = useRef(null);

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
                } flex items-center justify-between px-4 py-3 rounded-lg`}
                style={{
                    minWidth: "200px",
                    maxWidth: "500px",
                    width: messageRef.current
                        ? `${messageRef.current.offsetWidth + 80}px`
                        : "auto",
                }}
            >
                <div className="flex items-center gap-3">
                    {type === "success" ? (
                        <CheckCircleIcon className="w-6 h-6 text-white" />
                    ) : (
                        <XCircleIcon className="w-6 h-6 text-white" />
                    )}
                    <span ref={messageRef} className="text-white whitespace-nowrap">
                        {message}
                    </span>
                </div>
            </div>
        </div>
    );
};

export default Toast;
