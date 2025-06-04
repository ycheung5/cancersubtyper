import React from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { useNavigate } from 'react-router-dom';
import AuthForm from '../components/auth/AuthForm';
import { loginUser } from "../redux/authSlice";
import { showToast } from "../redux/toastSlice";
import { RouteConstants } from "../shared/constants/RouteConstants";

const Login = () => {
    const dispatch = useDispatch();
    const navigate = useNavigate();
    const { status } = useSelector((state) => state.auth); // `error` is already handled in `AuthForm`

    const handleLogin = (userData) => {
        dispatch(loginUser(userData))
            .unwrap()
            .then(() => {
                dispatch(showToast({ message: 'Login successful!', type: 'success' }));
                navigate(RouteConstants.home);
            })
            .catch((err) => {
                const errorMessage = typeof err === "string" ? err : "Login failed";
                dispatch(showToast({ message: errorMessage, type: 'error' }));
            });
    };

    return <AuthForm type="login" onSubmit={handleLogin} status={status} />;
};

export default Login;
