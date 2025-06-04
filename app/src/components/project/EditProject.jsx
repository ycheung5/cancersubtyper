import React, { useRef, useState, useEffect } from "react";
import { useDispatch, useSelector } from "react-redux";
import { updateProject } from "../../redux/projectSlice.jsx";
import { showToast } from "../../redux/toastSlice.jsx";

const EditProject = ({ isOpen, onClose, project }) => {
    const modalRef = useRef(null);
    const dispatch = useDispatch();
    const { status } = useSelector((state) => state.project);

    const [form, setForm] = useState({
        name: project?.name || "",
        description: project?.description || "",
        tumor_type: project?.tumor_type || "",
    });

    useEffect(() => {
        if (modalRef.current) {
            if (isOpen) {
                modalRef.current.showModal();
            } else {
                modalRef.current.close();
            }
        }
    }, [isOpen]);

    useEffect(() => {
        if (project) {
            setForm({
                name: project.name || "",
                description: project.description || "",
                tumor_type: project.tumor_type || "",
            });
        }
    }, [project]);

    const handleChange = (e) => {
        setForm({ ...form, [e.target.name]: e.target.value });
    };

    const handleSubmit = async (e) => {
        e.preventDefault();

        const changes = {};
        if (form.name !== project.name) changes.name = form.name;
        if (form.tumor_type !== project.tumor_type) changes.tumor_type = form.tumor_type;
        if (form.description !== project.description) changes.description = form.description;

        if (Object.keys(changes).length === 0) {
            dispatch(showToast({ message: "No changes made.", type: "info" }));
            onClose(false);
            return;
        }

        dispatch(updateProject({ projectId: project.id, projectData: changes }))
            .unwrap()
            .then(() => {
                dispatch(showToast({ message: "Project updated successfully!", type: "success" }));
                onClose(true);
            })
            .catch((err) => {
                dispatch(showToast({ message: err || "Failed to update project." , type: "error" }));
            });
    };

    const handleClose = () => {
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
                <h3 className="font-bold text-2xl text-primary mb-4">Edit Project</h3>

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

                    {/* Tumor Type Input */}
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
                            {status === "loading" ? "Saving..." : "Save Changes"}
                        </button>
                    </div>
                </form>
            </div>
        </dialog>
    );
};

export default EditProject;
