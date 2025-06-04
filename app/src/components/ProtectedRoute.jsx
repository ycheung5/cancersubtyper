import React, { useEffect } from "react";
import { useSelector } from "react-redux";
import { useNavigate } from "react-router-dom";
import { RouteConstants } from "../shared/constants/RouteConstants";

const ProtectedRoute = ({ children }) => {
    const navigate = useNavigate();
    const { token, status } = useSelector((state) => state.auth);

    useEffect(() => {
        if (!token && status !== "loading") {
            navigate(RouteConstants.login);
        }
    }, [token, status, navigate]);

    if (!token) {
        return <p className="text-center mt-20 text-error">Redirecting to login...</p>;
    }

    return children;
};

export default ProtectedRoute;
