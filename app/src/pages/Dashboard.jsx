import React from "react";
import StorageInfo from "../components/dashboard/StorageInfo.jsx";
import ProjectTable from "../components/dashboard/ProjectTable.jsx";
import BackToTopButton from "../components/BackToTopButton.jsx";

const Dashboard = () => {

    return (
        <div className="min-h-screen bg-base-100 p-10 flex flex-col items-center pt-[var(--navbar-height)] mt-5 overflow-auto">
            <h1 className="text-4xl font-bold mb-8 text-base-content flex items-center gap-2">
                Dashboard
            </h1>

            <StorageInfo />

            <ProjectTable />

            <BackToTopButton />
        </div>
    );
};

export default Dashboard;
