import React, { useRef, useState, useEffect } from "react";
import { useDispatch, useSelector } from "react-redux";
import { createProject } from "../../redux/projectSlice.jsx";
import { showToast } from "../../redux/toastSlice.jsx";

const CreateProjectModal = ({ isOpen, onClose }) => {
    const modalRef = useRef(null);
    const dispatch = useDispatch();
    const { status } = useSelector((state) => state.project);

    const [form, setForm] = useState({
        name: "",
        description: "",
        tumor_type: "",
    });

    // Open/close the modal when isOpen changes
    useEffect(() => {
        if (modalRef.current) {
            if (isOpen) {
                modalRef.current.showModal();
            } else {
                modalRef.current.close();
            }
        }
    }, [isOpen]);

    const handleChange = (e) => {
        setForm({ ...form, [e.target.name]: e.target.value });
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!form.name.trim() || !form.tumor_type) return;

        dispatch(createProject(form))
            .unwrap()
            .then(() => {
                dispatch(showToast({ message: "Project created successfully!", type: "success" }));
                setForm({ name: "", description: "", tumor_type: "" });
                onClose(true);
            })
            .catch(() => {
                dispatch(showToast({ message: "Failed to create project.", type: "error" }));
            });
    };

    const handleClose = () => {
        setForm({ name: "", description: "", tumor_type: "" });
        onClose(false);
    };

    return (
        <dialog
            ref={modalRef}
            className="modal"
            onCancel={(e) => e.preventDefault()}
        >
            <div
                className="modal-box bg-base-100 shadow-xl border border-base-300 rounded-xl p-6"
                onClick={(e) => e.stopPropagation()}
            >
                <h3 className="font-bold text-2xl text-primary mb-4">Create a New Project</h3>

                <form className="space-y-6" onSubmit={handleSubmit}>
                    {/* Project Name */}
                    <div className="form-control">
                        <label className="label">
                            <span className="label-text text-base-content font-medium mb-2">Project Name</span>
                        </label>
                        <input
                            type="text"
                            name="name"
                            value={form.name}
                            onChange={handleChange}
                            required
                            className="input input-bordered bg-base-200 text-base-content w-full"
                            placeholder="Enter project name..."
                        />
                    </div>

                    {/* Tumor Type Dropdown */}
                    <div className="form-control">
                        <label className="label">
                            <span className="label-text text-base-content font-medium mb-2">Tumor Type</span>
                        </label>
                        <input
                            type="text"
                            name="tumor_type"
                            value={form.tumor_type}
                            onChange={handleChange}
                            required
                            className="input input-bordered bg-base-200 text-base-content w-full"
                            placeholder="Enter tumor type..."
                        />
                    </div>

                    {/* Project Description */}
                    <div className="form-control">
                        <label className="label">
                            <span className="label-text text-base-content font-medium mb-2">Description</span>
                        </label>
                        <textarea
                            name="description"
                            value={form.description}
                            onChange={handleChange}
                            rows="5"
                            className="textarea textarea-bordered bg-base-200 text-base-content w-full resize-none"
                            placeholder="Describe your project..."
                        ></textarea>
                    </div>

                    {/* Modal Actions */}
                    <div className="modal-action flex justify-end gap-4">
                        <button
                            type="button"
                            className="btn btn-outline border-base-300 hover:border-base-400 text-base-content px-5"
                            onClick={handleClose}
                        >
                            Cancel
                        </button>
                        <button
                            type="submit"
                            className="btn btn-primary px-6 transition-all hover:shadow-md"
                            disabled={status === "loading" || !form.name.trim() || !form.tumor_type}
                        >
                            {status === "loading" ? "Creating..." : "Create Project"}
                        </button>
                    </div>
                </form>
            </div>
        </dialog>
    );
};

export default CreateProjectModal;
