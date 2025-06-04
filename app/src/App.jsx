import React, {useEffect} from 'react';
import {BrowserRouter, Routes, Route, Navigate} from 'react-router-dom';
import Home from './pages/Home';
import Login from './pages/Login';
import Signup from './pages/Signup';
import Toast from './components/Toast';
import {RouteConstants} from "./shared/constants/RouteConstants";
import Navbar from "./components/Navbar.jsx";
import {useDispatch, useSelector} from "react-redux";
import {fetchCurrentUser} from "./redux/authSlice.jsx";
import Dashboard from "./pages/Dashboard.jsx";
import ProtectedRoute from "./components/ProtectedRoute.jsx";
import Project from "./pages/Project.jsx";
import Visualization from "./pages/Visualization.jsx";

const App = () => {
    const dispatch = useDispatch();
    const user = useSelector(state => state.auth.user);

    useEffect(() => {
        if (!user) {
            dispatch(fetchCurrentUser());
        }
    }, [dispatch, user]);

    return (
        <BrowserRouter>
            <Navbar />
            <Toast />
            <Routes>
                <Route path={RouteConstants.home} element={<Home />} />
                <Route path={RouteConstants.login} element={<Login />} />
                <Route path={RouteConstants.signup} element={<Signup />} />
                <Route
                    path={RouteConstants.dashboard}
                    element={
                        <ProtectedRoute>
                            <Dashboard />
                        </ProtectedRoute>
                    }
                />
                <Route
                    path={`${RouteConstants.project}/:id`}
                    element={
                        <ProtectedRoute>
                            <Project />
                        </ProtectedRoute>
                    }
                />
                <Route
                    path={`${RouteConstants.project}/:project_id/${RouteConstants.visualization}/:job_id/:model_name`}
                    element={
                        <ProtectedRoute>
                            <Visualization />
                        </ProtectedRoute>
                    }
                />
                <Route path="*" element={<Navigate to={RouteConstants.home} replace />} />
            </Routes>
        </BrowserRouter>
    );
};

export default App;

