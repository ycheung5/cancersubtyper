import React, {useEffect, useRef, useState} from "react";
import { Link, useNavigate, useLocation } from "react-router-dom";
import { useDispatch, useSelector } from "react-redux";
import { logout } from "../redux/authSlice";
import { RouteConstants } from "../shared/constants/RouteConstants";
import icon from "../assets/icon.png";

const Navbar = () => {
    const { token } = useSelector((state) => state.auth);
    const dispatch = useDispatch();
    const navigate = useNavigate();
    const location = useLocation();
    const navRef = useRef(null);
    const [navHeight, setNavHeight] = useState(0);

    const [scrolled, setScrolled] = useState(false);

    useEffect(() => {
        const handleScroll = () => {
            if (location.pathname === RouteConstants.home) {
                setScrolled(window.scrollY > 80);
            } else {
                setScrolled(true);
            }
        };

        handleScroll();
        window.addEventListener("scroll", handleScroll);
        return () => window.removeEventListener("scroll", handleScroll);
    }, [location.pathname]);

    const handleLogout = () => {
        dispatch(logout());
        navigate(RouteConstants.login);
    };

    useEffect(() => {
        if (navRef.current) {
            setNavHeight(navRef.current.clientHeight);
        }
    }, [])

    return (
        <div>
            <nav
                ref={navRef}
                className={`fixed top-0 left-0 w-full transition-all duration-300 z-50 ${
                    scrolled ? "bg-base-300/100 shadow-lg" : "bg-transparent"
                }`}
            >
                <div className="container mx-auto px-6 py-4 flex justify-between items-center">
                    {/* Logo */}
                    <Link to={RouteConstants.home} onClick={() => setScrolled(false)} className="flex items-center space-x-2">
                        <img src={icon} alt="CancerSubtyper Logo" className="h-10 w-10 object-contain" />
                        <span className={`font-bold text-2xl tracking-wide transition-all duration-300 ${
                            scrolled ? "text-base-content" : "text-white"
                        }`}>
                            CancerSubtyper
                        </span>
                    </Link>


                    {/* Navigation Links */}
                    <div className="flex items-center gap-x-6">
                        {token ? (
                            <>
                                <button
                                    onClick={handleLogout}
                                    className={`btn px-5 py-2 rounded-lg transition-all duration-200 ${
                                        scrolled
                                            ? "btn-error opacity-80 text-white hover:btn-error-content"
                                            : "btn-white text-error hover:bg-error/50"
                                    }`}
                                >
                                    Logout
                                </button>
                            </>
                        ) : null}
                    </div>
                </div>
            </nav>
            {scrolled && <div style={{ paddingTop: `${navHeight}px` }}></div>}
        </div>
    );
};

export default Navbar;
