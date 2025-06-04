import React, { useEffect } from "react";
import { useSelector, useDispatch } from "react-redux";
import { AiOutlineDatabase } from "react-icons/ai";
import { getStorageInfo } from "../../redux/projectSlice.jsx";

const StorageInfo = () => {
    const dispatch = useDispatch();
    const { storageInfo } = useSelector((state) => state.project);

    useEffect(() => {
        dispatch(getStorageInfo()); // Fetch storage info on mount
    }, [dispatch]);

    return (
        <div className="bg-base-100 shadow-lg rounded-lg p-6 mb-8 border border-base-300 max-w-4xl w-full">
            <h2 className="text-xl font-semibold text-base-content mb-2 flex items-center gap-2">
                <AiOutlineDatabase className="text-primary text-2xl" />
                Storage Usage
            </h2>

            {!storageInfo ? (
                <div className="animate-pulse">
                    <div className="h-4 bg-base-300 rounded w-32 mb-2"></div>
                    <progress className="progress w-full mt-3"></progress>
                </div>
            ) : (
                <>
                    <p className="text-base-content text-lg">
                        Used:
                        <span className="font-bold text-primary"> {storageInfo.current_storage_usage} </span> /
                        <span className="font-bold"> {storageInfo.max_storage_limit} </span>
                    </p>
                    <progress
                        className="progress progress-primary w-full mt-3"
                        value={parseFloat(storageInfo.current_storage_usage)}
                        max={parseFloat(storageInfo.max_storage_limit)}
                    ></progress>
                </>
            )}
        </div>
    );
};

export default StorageInfo;
