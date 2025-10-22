import React from 'react';
import bcms from "../../assets/bctypefinder-model-structure.png";
import csms from "../../assets/cancersubminer-model-structure.png";
import nsf from "../../assets/nsf-logo.png";
import nih from "../../assets/nih-logo.png";
import webstructure from "../../assets/cancersubtyper-web-structure.png";

const ModelStructure = () => {
    return (
        <div id="model-structure" className="pt-20 pb-4 bg-gray-50">
            <div className="max-w-7xl mx-auto px-6 text-center">
                <h2 className="text-4xl font-extrabold mb-14 text-gray-800">
                Functionalities of CancerSubtyper (Data processing, deep learning modeling, visualization, and downstream analysis)
                </h2>
                <div className="grid grid-cols-1 gap-12">
                    <a href={webstructure} target="_blank" rel="noopener noreferrer">
                        <img
                            src={webstructure}
                            alt="Web Structure"
                            className="rounded-lg w-[90%] object-contain mx-auto cursor-pointer hover:opacity-90 transition"
                        />
                    </a>
                </div>

                <h2 className="text-4xl font-extrabold my-14 text-gray-800">
                Deep Learning-Based Cancer Subtyping Models in CancerSubtyper
                </h2>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-12">
                    {/* BCtypeFinder */}
                    <div className="bg-white p-6 rounded-2xl shadow-md hover:shadow-xl transition">
                        <h3 className="text-2xl font-semibold mb-6 text-red-500">
                            BCtypeFinder
                        </h3>
                        <a href={"https://www.liebertpub.com/doi/abs/10.1177/15578666251380233?casa_token=FrsS1pL4s9EAAAAA%3ALj5R2mvC-Fvwf6j1QgLUi7qvyXVJ65__j5N-VfGLV9uNG3bmpppJMNV-xpmgqKSOuVJIk5Svw50"} target="_blank" rel="noopener noreferrer">
                            <img
                                src={bcms}
                                alt="BCtypeFinder Model Structure"
                                className="rounded-lg w-full min-h-[350px] object-contain mx-auto cursor-pointer hover:opacity-90 transition"
                            />
                        </a>
                        <p className="text-sm text-gray-500 mt-4">
                        A cancer subtype prediction framework that utilizes a domain adaptation network combined with semi-supervised learning to address batch effects.
                        </p>
                    </div>

                    {/* CancerSubminer */}
                    <div className="bg-white p-6 rounded-2xl shadow-md hover:shadow-xl transition">
                        <h3 className="text-2xl font-semibold mb-6 text-red-500">
                            CancerSubminer
                        </h3>
                        <a href={"https://www.biorxiv.org/content/10.1101/2025.10.17.682936v1"} target="_blank" rel="noopener noreferrer">
                            <img
                                src={csms}
                                alt="CancerSubminer Model Structure"
                                className="rounded-lg w-full min-h-[350px] object-contain mx-auto cursor-pointer hover:opacity-90 transition"
                            />
                        </a>
                        <p className="text-sm text-gray-500 mt-4">
                        An integrative sub typing framework that combines supervised and unsupervised learning to facilitate the discovery of novel subgroups in less-characterized cancer.
                        </p>
                    </div>
                </div>
            </div>
            <div>
                <p className="text-md font-bold text-gray-600 mt-12 text-center">
                This work was supported by the U.S. National Science Foundation (NSF) under Awards #2004751, #2125798, #2344169, and #2319522, as well as the National Institutes of Health (NIH) grant #1R01AI179686-01A1.
                </p>
                <div className="flex justify-center items-center gap-10 mt-4">
                    <img src={nsf} alt="NSF Logo" className="w-24" />
                    <img src={nih} alt="NIH Logo" className="w-24" />
                </div>
            </div>
        </div>
    );
};

export default ModelStructure;
