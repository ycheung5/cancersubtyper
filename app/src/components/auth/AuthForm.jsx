import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import authBg from '../../assets/hero-bg.png';

const timezones = [
    "UTC", "America/New_York", "America/Chicago", "America/Denver", "America/Los_Angeles",
    "Europe/London", "Europe/Paris", "Asia/Tokyo", "Asia/Hong_Kong", "Asia/Seoul"
];

const AuthForm = ({ type, onSubmit, status, error }) => {
    const [form, setForm] = useState({
        username: '',
        email: '',
        password: '',
        confirmPassword: '',
        timezone: Intl.DateTimeFormat().resolvedOptions().timeZone || "UTC",
    });

    const [validationError, setValidationError] = useState('');

    useEffect(() => {
        setValidationError('');
    }, [form.password, form.confirmPassword]);

    const handleChange = (e) => {
        setForm({ ...form, [e.target.name]: e.target.value });
    };

    const handleSubmit = (e) => {
        e.preventDefault();
        if (type === "signup" && form.password !== form.confirmPassword) {
            setValidationError("Passwords do not match");
            return;
        }

        const userData = type === "signup"
            ? { username: form.username, email: form.email, password: form.password, timezone: form.timezone }
            : { username: form.username, password: form.password };

        onSubmit(userData);
    };

    return (
        <div
            className="flex items-center justify-center min-h-screen bg-cover bg-center"
            style={{ backgroundImage: `url(${authBg})` }} // Set Background Image
        >
            <div className="bg-base-200 shadow-xl w-full max-w-md p-8 rounded-2xl border border-base-content/20">
                <h2 className="text-3xl font-bold text-center text-base-content mb-6 tracking-wide">
                    {type === 'signup' ? 'Create Account' : 'Login'}
                </h2>

                {/* Error Messages */}
                {error && (
                    <div className="alert alert-error text-base-content p-2 rounded-md mb-4">
                        {error}
                    </div>
                )}
                {validationError && (
                    <div className="alert alert-warning text-base-content p-2 rounded-md mb-4">
                        {validationError}
                    </div>
                )}

                <form onSubmit={handleSubmit} className="space-y-5">
                    <div className="form-control w-full">
                        <label className="input input-bordered flex items-center gap-2 bg-base-300 text-base-content w-full rounded-lg">
                            <input
                                type="text"
                                name="username"
                                placeholder="Username"
                                className="w-full bg-transparent outline-none text-base-content placeholder-base-content/50"
                                value={form.username}
                                onChange={handleChange}
                                required
                            />
                        </label>
                    </div>

                    {type === 'signup' && (
                        <>
                            <div className="form-control w-full">
                                <label className="input input-bordered flex items-center gap-2 bg-base-300 text-base-content w-full rounded-lg">
                                    <input
                                        type="email"
                                        name="email"
                                        placeholder="Email"
                                        className="w-full bg-transparent outline-none text-base-content placeholder-base-content/50"
                                        value={form.email}
                                        onChange={handleChange}
                                        required
                                    />
                                </label>
                            </div>

                            <div className="form-control w-full">
                                <label className="input input-bordered flex items-center gap-2 bg-base-300 text-base-content w-full rounded-lg">
                                    <select
                                        name="timezone"
                                        className="bg-transparent outline-none text-base-content w-full"
                                        value={form.timezone}
                                        onChange={handleChange}
                                        required
                                    >
                                        {timezones.map((tz) => (
                                            <option key={tz} value={tz} className="text-base-content">
                                                {tz}
                                            </option>
                                        ))}
                                    </select>
                                </label>
                            </div>
                        </>
                    )}

                    <div className="form-control w-full">
                        <label className="input input-bordered flex items-center gap-2 bg-base-300 text-base-content w-full rounded-lg">
                            <input
                                type="password"
                                name="password"
                                placeholder="Password"
                                className="w-full bg-transparent outline-none text-base-content placeholder-base-content/50"
                                value={form.password}
                                onChange={handleChange}
                                required
                            />
                        </label>
                    </div>

                    {type === 'signup' && (
                        <div className="form-control w-full">
                            <label className="input input-bordered flex items-center gap-2 bg-base-300 text-base-content w-full rounded-lg">
                                <input
                                    type="password"
                                    name="confirmPassword"
                                    placeholder="Confirm Password"
                                    className="w-full bg-transparent outline-none text-base-content placeholder-base-content/50"
                                    value={form.confirmPassword}
                                    onChange={handleChange}
                                    required
                                />
                            </label>
                        </div>
                    )}

                    {/* ✅ Stylish Button with Spinner */}
                    <button
                        type="submit"
                        className="btn btn-primary w-full mt-4 flex justify-center items-center gap-2 text-lg font-semibold shadow-lg transition-transform transform active:scale-95"
                        disabled={status === 'loading'}
                    >
                        {status === 'loading' ? (
                            <>
                                <span className="loading loading-spinner"></span>
                                Processing...
                            </>
                        ) : type === 'signup' ? 'Sign Up' : 'Login'}
                    </button>

                    <p className="text-center text-base-content mt-4">
                        {type === 'signup' ? (
                            <>
                                Already have an account?{' '}
                                <Link to="/login" className="text-primary hover:underline">
                                    Login here
                                </Link>
                            </>
                        ) : (
                            <>
                                Don't have an account?{' '}
                                <Link to="/signup" className="text-primary hover:underline">
                                    Sign up here
                                </Link>
                            </>
                        )}
                    </p>
                </form>
            </div>
        </div>
    );
};

export default AuthForm;
