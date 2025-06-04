import React from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { useNavigate } from 'react-router-dom';
import AuthForm from '../components/auth/AuthForm';
import { signupUser } from "../redux/authSlice";  // FIX: Correct import
import { showToast } from "../redux/toastSlice";
import { RouteConstants } from "../shared/constants/RouteConstants";

const Signup = () => {
    const dispatch = useDispatch();
    const navigate = useNavigate();
    const { status } = useSelector((state) => state.auth); // `error` is already handled in `AuthForm`

    const handleSignup = (userData) => {
        dispatch(signupUser(userData))
            .unwrap()
            .then(() => {
                dispatch(showToast({ message: 'Signup successful!', type: 'success' }));
                navigate(RouteConstants.dashboard);
            })
            .catch((err) => {
                const errorMessage = typeof err === "string" ? err : "Signup failed";
                dispatch(showToast({ message: errorMessage, type: 'error' }));
            });
    };

    return <AuthForm type="signup" onSubmit={handleSignup} status={status} />;
};

export default Signup;
