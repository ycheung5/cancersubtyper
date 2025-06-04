import React, { useState, useEffect } from "react";
import { FaArrowUp } from "react-icons/fa";

const BackToTopButton = () => {
    const [isVisible, setIsVisible] = useState(false);

    // Handle scroll event
    useEffect(() => {
        const toggleVisibility = () => {
            setIsVisible(window.scrollY > 300);
        };

        window.addEventListener("scroll", toggleVisibility);
        return () => window.removeEventListener("scroll", toggleVisibility);
    }, []);

    // Scroll to top function
    const scrollToTop = () => {
        window.scrollTo({ top: 0, behavior: "smooth" });
    };

    return (
        <button
            onClick={scrollToTop}
            className={`btn btn-primary fixed bottom-6 right-6 p-3 rounded-full shadow-lg transition-opacity ${
                isVisible ? "opacity-100" : "opacity-0 pointer-events-none"
            }`}
            aria-label="Back to Top"
        >
            <FaArrowUp className="text-xl" />
        </button>
    );
};

export default BackToTopButton;
