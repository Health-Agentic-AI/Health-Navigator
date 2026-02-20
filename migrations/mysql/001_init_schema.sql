-- Health Navigator MySQL schema (matches SQLAlchemy models in app/models.py and app/compliance.py)
-- Target: MySQL 8.0+
-- Usage:
--   mysql -u <user> -p < migrations/mysql/001_init_schema.sql
--
-- If you use a different DB name, replace `medical_db` below.

CREATE DATABASE IF NOT EXISTS `medical_db`
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE `medical_db`;

SET NAMES utf8mb4;
SET time_zone = '+00:00';

-- ----------------------------
-- Core domain tables
-- ----------------------------

CREATE TABLE IF NOT EXISTS `users` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `full_name` VARCHAR(255) NOT NULL,
  `username` VARCHAR(255) NOT NULL,
  `email` VARCHAR(255) NOT NULL,
  `password_hash` VARCHAR(255) NOT NULL,
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uq_users_username` (`username`),
  UNIQUE KEY `uq_users_email` (`email`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS `patient_profiles` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `user_id` INT NOT NULL,
  `age` INT NULL,
  `gender` VARCHAR(20) NULL,
  `occupation` VARCHAR(255) NULL,
  `is_smoker` TINYINT(1) NULL DEFAULT 0,
  `smoking_details` TEXT NULL,
  `alcohol_consumption` VARCHAR(50) NULL,
  `alcohol_details` TEXT NULL,
  `socioeconomic_status` VARCHAR(50) NULL,
  `last_updated_at` DATETIME NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uq_patient_profiles_user_id` (`user_id`),
  CONSTRAINT `fk_patient_profiles_user_id`
    FOREIGN KEY (`user_id`) REFERENCES `users` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS `conversations` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `user_id` INT NOT NULL,
  `title` VARCHAR(255) NULL,
  `started_at` DATETIME NULL DEFAULT CURRENT_TIMESTAMP,
  `last_updated_at` DATETIME NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `ix_conversations_user_id` (`user_id`),
  CONSTRAINT `fk_conversations_user_id`
    FOREIGN KEY (`user_id`) REFERENCES `users` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS `messages` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `conversation_id` INT NOT NULL,
  `sender_type` VARCHAR(50) NOT NULL,
  `content` TEXT NOT NULL,
  `created_at` DATETIME NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `ix_messages_conversation_id` (`conversation_id`),
  CONSTRAINT `fk_messages_conversation_id`
    FOREIGN KEY (`conversation_id`) REFERENCES `conversations` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS `attachments` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `message_id` INT NOT NULL,
  `file_path` TEXT NOT NULL,
  `file_type` VARCHAR(50) NULL,
  `original_name` VARCHAR(255) NULL,
  `uploaded_at` DATETIME NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `ix_attachments_message_id` (`message_id`),
  CONSTRAINT `fk_attachments_message_id`
    FOREIGN KEY (`message_id`) REFERENCES `messages` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS `allergies` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `user_id` INT NOT NULL,
  `allergy_name` VARCHAR(255) NOT NULL,
  `allergy_type` VARCHAR(50) NULL,
  `severity` VARCHAR(50) NULL,
  `notes` TEXT NULL,
  `created_at` DATETIME NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `ix_allergies_user_id` (`user_id`),
  CONSTRAINT `fk_allergies_user_id`
    FOREIGN KEY (`user_id`) REFERENCES `users` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS `medications` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `user_id` INT NOT NULL,
  `medication_name` VARCHAR(255) NOT NULL,
  `dosage` VARCHAR(100) NULL,
  `frequency` VARCHAR(100) NULL,
  `started_at` DATE NULL,
  `ended_at` DATE NULL,
  `is_current` TINYINT(1) NULL DEFAULT 1,
  `notes` TEXT NULL,
  `created_at` DATETIME NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `ix_medications_user_id` (`user_id`),
  CONSTRAINT `fk_medications_user_id`
    FOREIGN KEY (`user_id`) REFERENCES `users` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS `past_medical_history` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `user_id` INT NOT NULL,
  `condition` VARCHAR(255) NOT NULL,
  `diagnosed_date` DATE NULL,
  `notes` TEXT NULL,
  `created_at` DATETIME NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `ix_past_medical_history_user_id` (`user_id`),
  CONSTRAINT `fk_past_medical_history_user_id`
    FOREIGN KEY (`user_id`) REFERENCES `users` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS `past_surgeries` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `user_id` INT NOT NULL,
  `surgery_name` VARCHAR(255) NOT NULL,
  `surgery_date` DATE NULL,
  `hospital` VARCHAR(255) NULL,
  `notes` TEXT NULL,
  `created_at` DATETIME NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `ix_past_surgeries_user_id` (`user_id`),
  CONSTRAINT `fk_past_surgeries_user_id`
    FOREIGN KEY (`user_id`) REFERENCES `users` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS `family_history` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `user_id` INT NOT NULL,
  `relation` VARCHAR(50) NULL,
  `condition` VARCHAR(255) NOT NULL,
  `age_of_diagnosis` INT NULL,
  `notes` TEXT NULL,
  `created_at` DATETIME NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `ix_family_history_user_id` (`user_id`),
  CONSTRAINT `fk_family_history_user_id`
    FOREIGN KEY (`user_id`) REFERENCES `users` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ----------------------------
-- Compliance tables
-- ----------------------------

CREATE TABLE IF NOT EXISTS `audit_logs` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `user_id` INT NULL,
  `action` VARCHAR(100) NOT NULL,
  `resource_type` VARCHAR(50) NULL,
  `resource_id` INT NULL,
  `ip_address` VARCHAR(45) NULL,
  `user_agent` VARCHAR(500) NULL,
  `status` VARCHAR(20) NULL DEFAULT 'success',
  `details` JSON NULL,
  `timestamp` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `ix_audit_logs_user_id` (`user_id`),
  KEY `ix_audit_logs_action` (`action`),
  KEY `ix_audit_logs_timestamp` (`timestamp`),
  CONSTRAINT `fk_audit_logs_user_id`
    FOREIGN KEY (`user_id`) REFERENCES `users` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS `consents` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `user_id` INT NOT NULL,
  `consent_type` VARCHAR(50) NOT NULL,
  `granted` TINYINT(1) NOT NULL DEFAULT 0,
  `granted_at` DATETIME NULL,
  `revoked_at` DATETIME NULL,
  `version` VARCHAR(20) NULL DEFAULT '1.0',
  `ip_address` VARCHAR(45) NULL,
  `consent_metadata` JSON NULL,
  `created_at` DATETIME NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` DATETIME NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uq_consents_user_id` (`user_id`),
  KEY `ix_consents_user_id` (`user_id`),
  KEY `ix_consents_consent_type` (`consent_type`),
  CONSTRAINT `fk_consents_user_id`
    FOREIGN KEY (`user_id`) REFERENCES `users` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS `data_retention_policies` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `data_type` VARCHAR(50) NOT NULL,
  `retention_period_days` INT NOT NULL,
  `deletion_action` VARCHAR(50) NULL DEFAULT 'hard_delete',
  `policy_description` TEXT NULL,
  `created_at` DATETIME NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` DATETIME NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uq_data_retention_policies_data_type` (`data_type`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
