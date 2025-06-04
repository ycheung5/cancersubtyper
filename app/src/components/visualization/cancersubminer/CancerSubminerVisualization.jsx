import React from 'react';
import Figure1 from "./figure/Figure1.jsx";
import Figure2 from "./figure/Figure2.jsx";
import Figure3 from "./figure/Figure3.jsx";
import Figure4 from "./figure/Figure4.jsx";
import Figure5 from "./figure/Figure5.jsx";

const CancerSubminerVisualization = ({ id }) => {
    return (
        <div className="space-y-6">
            <h2 className="text-2xl font-semibold text-center text-base-content">
                CancerSubminer Visualization - Job ID: {id}
            </h2>

            <div className="grid gap-6">
                <Figure1 id={id} />
                <Figure2 id={id} />
                <Figure3 id={id} />
                <Figure4 id={id} />
                <Figure5 id={id} />
            </div>
        </div>
    );
};

export default CancerSubminerVisualization;