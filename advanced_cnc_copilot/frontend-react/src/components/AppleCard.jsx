import React from 'react';
import { motion } from 'framer-motion';

const AppleCard = ({ children, className = "", title, subtitle, icon: Icon, action }) => {
    return (
        <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, ease: "easeOut" }}
            className={`apple-card p-6 flex flex-col ${className}`}
        >
            {(title || Icon) && (
                <div className="flex justify-between items-start mb-4">
                    <div className="flex items-center gap-3">
                        {Icon && (
                            <div className="w-10 h-10 rounded-full bg-gray-50 flex items-center justify-center text-[var(--apple-text-primary)] shadow-sm">
                                <Icon size={20} />
                            </div>
                        )}
                        <div>
                            {title && <h3 className="text-[17px] font-semibold text-[var(--apple-text-primary)] leading-tight">{title}</h3>}
                            {subtitle && <p className="text-[13px] text-[var(--apple-text-secondary)] font-medium">{subtitle}</p>}
                        </div>
                    </div>
                    {action && <div>{action}</div>}
                </div>
            )}
            <div className="flex-1 relative">
                {children}
            </div>
        </motion.div>
    );
};

export default AppleCard;
