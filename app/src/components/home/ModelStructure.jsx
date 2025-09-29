import React from 'react';
import bcms from "../../assets/bctypefinder-model-structure.png";
import csms from "../../assets/cancersubminer-model-structure.png";

const ModelStructure = () => {
    return (
        <div id="model-structure" className="py-20 bg-gray-50">
            <div className="max-w-7xl mx-auto px-6 text-center">
                <h2 className="text-4xl font-extrabold mb-14 text-gray-800">
                    Model Infrastructure
                </h2>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-12">
                    {/* BCtypeFinder */}
                    <div className="bg-white p-6 rounded-2xl shadow-md hover:shadow-xl transition">
                        <h3 className="text-2xl font-semibold mb-6 text-red-500">
                            BCtypeFinder
                        </h3>
                        <a href={bcms} target="_blank" rel="noopener noreferrer">
                            <img
                                src={bcms}
                                alt="BCtypeFinder Model Structure"
                                className="rounded-lg w-full max-h-[700px] object-contain mx-auto cursor-pointer hover:opacity-90 transition"
                            />
                        </a>
                    </div>

                    {/* CancerSubminer */}
                    <div className="bg-white p-6 rounded-2xl shadow-md hover:shadow-xl transition">
                        <h3 className="text-2xl font-semibold mb-6 text-red-500">
                            CancerSubminer
                        </h3>
                        <a href={csms} target="_blank" rel="noopener noreferrer">
                            <img
                                src={csms}
                                alt="CancerSubminer Model Structure"
                                className="rounded-lg w-full max-h-[700px] object-contain mx-auto cursor-pointer hover:opacity-90 transition"
                            />
                        </a>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default ModelStructure;
